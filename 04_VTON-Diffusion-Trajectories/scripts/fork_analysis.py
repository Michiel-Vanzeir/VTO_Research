"""
Fork experiment analysis: WHEN is each property of the try-on decided?

Input: outputs/_forks/<model>_<pair>/k<KK>_s<S>.png, written by the model
scripts in fork mode (--fork-steps, see common.py). For fork step k, every
fork with that k shares the exact denoising path up to step k and then
continues with its own noise. Whatever the forks still agree on at the end
was already decided at step k; whatever differs was still open.

Per (model, pair, k) and per garment axis (the validated metrics.py axes,
same mask and resolution) we compute the mean agreement over all pairs of
forks, using scores_vs_reference with one fork as the reference for the
other (all four axes are symmetric):

  agreement(k)  in [0, 1], 1 = forks identical on that axis.
  decided(k)    = (agreement(k) - agreement(0)) / (1 - agreement(0)):
                  0 = no more agreement than fully independent samples
                  (k = 0 forks also draw different initial noise),
                  1 = fully decided. Because every axis is normalized to its
                  OWN baseline and ceiling, decided() is comparable ACROSS
                  axes within one model -- unlike the final-referenced curves
                  of metrics.py, whose axes start at different levels.
  decision_step the (linearly interpolated) k at which decided reaches 0.5:
                  "half-decided". NaN if never reached by the last fork step.

Pose is left out: every model has its pose fixed from the first steps
(PCK vs final ~0.98 at step 1), so there is nothing to fork on.

Outputs in outputs/_analysis/forks/:
  per_fork_step.csv     model, pair, category, k, <axis>_agreement, <axis>_decided
  decision_steps.csv    model, pair, category, <axis>_decision_step
  summary.txt           mean decision step per model/axis + paired Wilcoxon tests
  decided_by_axis.png   one panel per axis, decided(k) per model (mean +- sem)
  decided_by_model.png  one panel per model, decided(k) per axis
  sheets/<model>_<pair>.jpg  contact sheet: rows = fork step, columns = seeds
"""
import argparse
import csv
import itertools
import json
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from compare_trajectories import MODELS, load_categories, parse_run  # noqa: E402
from metrics import (  # noqa: E402
    ANALYSIS_SIZE, Reference, garment_mask_path, load_mask, load_rgb, resolve_input, scores_vs_reference,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FORKS_ROOT = REPO_ROOT / "outputs" / "_forks"
OUT = REPO_ROOT / "outputs" / "_analysis" / "forks"
AXES = ("color", "structure", "texture", "pattern")
AXIS_COLORS = {"color": "#10b981", "structure": "#f59e0b", "texture": "#f97316", "pattern": "#8b5cf6"}
DECISION_LEVEL = 0.5
FORK_RE = re.compile(r"k(\d+)_s(\d+)\.png")


def load_forks(run_dir: Path) -> dict:
    """k -> list of fork image paths."""
    forks = {}
    for p in sorted(run_dir.glob("k*_s*.png")):
        m = FORK_RE.fullmatch(p.name)
        if m:
            forks.setdefault(int(m.group(1)), []).append(p)
    return forks


def agreement(images, mask) -> dict:
    """Mean per-axis score over all unordered pairs of fork images."""
    scores = {a: [] for a in AXES}
    refs = [Reference(img, mask) for img in images]
    for i, j in itertools.combinations(range(len(images)), 2):
        for a, v in scores_vs_reference(images[i], refs[j]).items():
            scores[a].append(v)
    return {a: float(np.mean(v)) for a, v in scores.items()}


def decision_step(ks, decided) -> float:
    """First k at which decided reaches DECISION_LEVEL, linearly interpolated."""
    for (k0, d0), (k1, d1) in zip(zip(ks, decided), zip(ks[1:], decided[1:])):
        if d0 >= DECISION_LEVEL:
            return float(k0)
        if d1 >= DECISION_LEVEL:
            return float(k0 + (DECISION_LEVEL - d0) / (d1 - d0) * (k1 - k0))
    return float(ks[0]) if decided and decided[0] >= DECISION_LEVEL else float("nan")


def analyze_run(run_dir: Path) -> dict:
    config = json.loads((run_dir / "fork_config.json").read_text())
    person = resolve_input(config["inputs"]["person"])
    mask = load_mask(garment_mask_path(person), ANALYSIS_SIZE)
    forks = load_forks(run_dir)
    ks = sorted(forks)
    assert ks and ks[0] == 0, f"{run_dir}: need a k=0 baseline, have fork steps {ks}"
    agree = {k: agreement([load_rgb(p) for p in forks[k]], mask) for k in ks}
    decided = {a: [(agree[k][a] - agree[0][a]) / max(1 - agree[0][a], 1e-6) for k in ks] for a in AXES}
    return {"ks": ks, "agreement": agree, "decided": decided, "forks": forks,
            "decision_step": {a: decision_step(ks, decided[a]) for a in AXES}}


def contact_sheet(forks: dict, out_path: Path, thumb=(192, 256)):
    ks = sorted(forks)
    n_cols = max(len(v) for v in forks.values())
    label_w = 48
    sheet = Image.new("RGB", (label_w + n_cols * thumb[0], len(ks) * thumb[1]), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    for r, k in enumerate(ks):
        draw.text((6, r * thumb[1] + thumb[1] // 2), f"k={k}", fill="black")
        for c, p in enumerate(forks[k]):
            sheet.paste(Image.open(p).convert("RGB").resize(thumb, Image.LANCZOS), (label_w + c * thumb[0], r * thumb[1]))
    sheet.save(out_path, quality=88)


def mean_sem(arr):
    arr = np.asarray(arr, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(arr, axis=0)
        n = np.sum(np.isfinite(arr), axis=0)
        s = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
    return m, s


def plot_panels(results, names, out_path, by):
    """by='axis': panel per axis, line per model; by='model': panel per model, line per axis."""
    panels, lines = (AXES, names) if by == "axis" else (names, AXES)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), sharey=True, squeeze=False)
    for ax, panel in zip(axes[0], panels):
        for line in lines:
            model, axis = (line, panel) if by == "axis" else (panel, line)
            runs = [r for (m, _), r in results.items() if m == model]
            if not runs:
                continue
            ks = runs[0]["ks"]
            runs = [r for r in runs if r["ks"] == ks]
            m, s = mean_sem([r["decided"][axis] for r in runs])
            color = dict(MODELS.values())[model] if by == "axis" else AXIS_COLORS[axis]
            ax.plot(ks, m, "o-", color=color, label=f"{line} (n={len(runs)})", markersize=4)
            ax.fill_between(ks, m - s, m + s, color=color, alpha=0.2)
        ax.axhline(DECISION_LEVEL, color="gray", linewidth=0.6, linestyle="--")
        ax.set_title(panel.upper() if by == "axis" else panel, fontsize=10)
        ax.set_xlabel("fork step k")
        ax.set_ylim(-0.1, 1.05)
        ax.legend(fontsize=7, loc="lower right")
    axes[0][0].set_ylabel("decided (0 = independent, 1 = identical)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--no-sheets", action="store_true", help="skip the per-run contact sheets")
    p.add_argument("--forks-root", type=Path, default=FORKS_ROOT)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    out = args.out

    categories_of = load_categories()
    runs = sorted(d for d in args.forks_root.glob("*_*") if (d / "fork_config.json").exists() and parse_run(d))
    if not runs:
        raise SystemExit(f"no finished fork runs (fork_config.json) under {args.forks_root}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "sheets").mkdir(exist_ok=True)

    results = {}
    for i, d in enumerate(runs):
        prefix, pair = parse_run(d)
        results[(MODELS[prefix][0], pair)] = r = analyze_run(d)
        if not args.no_sheets:
            contact_sheet(r["forks"], out / "sheets" / f"{d.name}.jpg")
        print(f"[forks] {i + 1}/{len(runs)} {d.name}", end="\r", flush=True)
    print()
    names = [v[0] for v in MODELS.values() if any(m == v[0] for m, _ in results)]

    with open(out / "per_fork_step.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "pair", "category", "k"] + [f"{a}_{s}" for a in AXES for s in ("agreement", "decided")])
        for (model, pair), r in results.items():
            for i, k in enumerate(r["ks"]):
                w.writerow([model, pair, categories_of.get(pair, "uncategorized"), k]
                           + [f"{v:.4f}" for a in AXES for v in (r["agreement"][k][a], r["decided"][a][i])])
    with open(out / "decision_steps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "pair", "category"] + [f"{a}_decision_step" for a in AXES])
        for (model, pair), r in results.items():
            w.writerow([model, pair, categories_of.get(pair, "uncategorized")]
                       + [f"{r['decision_step'][a]:.2f}" for a in AXES])

    lines = [f"Fork experiment: {len(results)} runs; decision step = first k with decided >= {DECISION_LEVEL}",
             "(lower = decided earlier; 'never' = not half-decided by the last fork step)", ""]
    for model in names:
        rs = [r for (m, _), r in results.items() if m == model]
        cells = []
        for a in AXES:
            v = np.array([r["decision_step"][a] for r in rs])
            fin = v[np.isfinite(v)]
            cells.append(f"{a} {np.median(fin):5.1f} (never {np.sum(~np.isfinite(v))}/{len(v)})" if len(fin)
                         else f"{a}   never")
        lines.append(f"{model:14} median decision step: " + " | ".join(cells))
    lines += ["", "Paired Wilcoxon between models (same pairs, 'never' counted as the last fork step + 10):"]
    for a in AXES:
        for m1, m2 in itertools.combinations(names, 2):
            pairs = sorted({p for m, p in results if m == m1} & {p for m, p in results if m == m2})
            if len(pairs) < 5:
                continue

            def ds(m):
                r = [results[(m, p)] for p in pairs]
                return np.array([x["decision_step"][a] if np.isfinite(x["decision_step"][a]) else x["ks"][-1] + 10
                                 for x in r])
            x, y = ds(m1), ds(m2)
            p_val = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
            lines.append(f"  {a:9} {m1} vs {m2}: median diff {np.median(x - y):+5.1f} steps, "
                         f"{m1} earlier in {np.mean(x < y):.0%} of {len(pairs)} pairs, p = {p_val:.2g}")
    lines += ["", "Within-model axis order (paired over pairs, Wilcoxon):"]
    for model in names:
        rs = [r for (m, _), r in results.items() if m == model]
        for a1, a2 in itertools.combinations(AXES, 2):
            x = np.array([r["decision_step"][a1] if np.isfinite(r["decision_step"][a1]) else r["ks"][-1] + 10 for r in rs])
            y = np.array([r["decision_step"][a2] if np.isfinite(r["decision_step"][a2]) else r["ks"][-1] + 10 for r in rs])
            if len(rs) < 5:
                continue
            p_val = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
            lines.append(f"  {model:14} {a1} vs {a2}: median diff {np.median(x - y):+5.1f}, p = {p_val:.2g}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plot_panels(results, names, out / "decided_by_axis.png", by="axis")
    plot_panels(results, names, out / "decided_by_model.png", by="model")
    print(f"[forks] done: see {out}")


if __name__ == "__main__":
    main()
