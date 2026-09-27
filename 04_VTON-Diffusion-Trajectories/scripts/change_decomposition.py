"""
WITHIN-model analysis: what does each model change when, during denoising?

Complements metrics.py (which compares axes whose scales differ, so their
order within one model can't be read off directly). Here every quantity is
decomposed the same way for every scale band, so bands CAN be compared
within a model:

  bands     each CIE Lab image is split into three spatial-scale bands that
            sum back to the image:
              coarse = G(8) * x               (> ~8 px: layout, large shading)
              mid    = G(2) * x - G(8) * x    (~2-8 px: pattern elements)
              fine   = x - G(2) * x           (< ~2 px: fine detail / edges)
            with G(s) a Gaussian blur of sigma s px at 384x512; plus a
            luminance (L) vs chroma (a, b) split.

Two views, both restricted to the garment region (softened dataset
agnostic mask, the same for every model):

  1. change shares (reference-free): of the change between consecutive
     x0-hat estimates, which share lies in each band -- "what is the model
     editing right now". Shares are proportions, not scores: higher only
     means "more of this step's change is in this band".
  2. band progress (in [0, 1], higher = further along): 1 - d_b(t) / d_b(1),
     d_b(t) = distance of band b to the final frame. Each band is normalized
     by its OWN starting distance, so all bands start at 0 and end at 1 and
     can be compared directly. Step 1 (not 0) is the start for all models:
     IDM-VTON's step-0 estimate is nearly pure noise.

Per-run summaries (higher = earlier, as everywhere else):
  <band>_auc                 mean progress over the steps
  coarse_to_fine_index       coarse_auc - fine_auc: > 0 means coarse scales
                             are resolved before fine ones (coarse-to-fine),
                             < 0 fine before coarse, ~0 all at once
  luma_before_chroma_index   L_auc - chroma_auc: > 0 means brightness
                             structure is resolved before color
  detail_phase_headstart     share of the trajectory still left when the
                             model enters its "detail phase": the first step
                             from which the fine band is (3-step average)
                             >= DETAIL_PHASE_SHARE of every later step's
                             change. 1 = from the start, 0 = never.

Expect every model to be coarse-to-fine to some degree: noise drowns fine
scales first, so diffusion models generally fill them in last ("spectral
autoregression"). The model differences are in WHEN the detail phase starts
and how gradually the bands hand over.

Writes per run: <run>/change_decomposition.json, and in
outputs/_analysis/change_decomposition/:
  per_run.csv, by_model.csv, by_category.csv, stats.txt,
  progress_by_band.png, change_shares.png, index_by_category.png

Usage:
    python scripts/change_decomposition.py
"""
import argparse
import csv
import json
import statistics
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon
from skimage.color import rgb2lab

from compare_trajectories import MODELS, load_categories, parse_run
from metrics import ANALYSIS_SIZE, garment_mask_path, load_mask, load_rgb, resolve_input

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION = 2
DETAIL_PHASE_SHARE = 0.5
FINE_SIGMA, COARSE_SIGMA = 2.0, 8.0
BANDS = ("coarse", "mid", "fine")
CHANNELS = ("L", "chroma")
START = 1  # first step used as the progress baseline (see module docstring)
BAND_COLORS = {"coarse": "#1d4ed8", "mid": "#7c3aed", "fine": "#db2777", "L": "#525252", "chroma": "#16a34a"}


def blur_lab(lab: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack([gaussian_filter(lab[..., c], sigma) for c in range(3)], -1)


def split_bands(lab: np.ndarray) -> dict:
    g_fine, g_coarse = blur_lab(lab, FINE_SIGMA), blur_lab(lab, COARSE_SIGMA)
    return {"coarse": g_coarse, "mid": g_fine - g_coarse, "fine": lab - g_fine}


def energy(x: np.ndarray, window: np.ndarray, channels=slice(None)) -> float:
    return float((window[..., None] * x[..., channels] ** 2).sum())


def detail_phase_headstart(fine_shares) -> float:
    """Share of the trajectory left when the fine band starts to make up
    >= DETAIL_PHASE_SHARE of the change at every remaining step (3-step
    moving average, edges padded with their own value -- zero padding would
    drag the last step below the threshold)."""
    fine = np.nan_to_num(np.asarray(fine_shares[1:], dtype=np.float64), nan=0.0)  # [0] is undefined (no previous step)
    smooth = np.convolve(np.pad(fine, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    n_steps = len(fine_shares)
    start = next((i + 1 for i in range(len(smooth)) if (smooth[i:] >= DETAIL_PHASE_SHARE).all()), None)
    return 0.0 if start is None else 1.0 - start / (n_steps - 1)


def summarize(progress: dict, shares: dict) -> dict:
    auc = {k: float(np.nanmean(v)) for k, v in progress.items()}
    return {**{f"{k}_auc": v for k, v in auc.items()},
            "coarse_to_fine_index": auc["coarse"] - auc["fine"],
            "luma_before_chroma_index": auc["L"] - auc["chroma"],
            "detail_phase_headstart": detail_phase_headstart(shares["fine"])}


def analyze_run(run_dir: Path) -> dict:
    """The expensive per-step curves are cached in <run>/change_decomposition.json;
    the summary is always recomputed from them (cheap), so changing a summary
    definition never requires re-decomposing the frames."""
    cache = run_dir / "change_decomposition.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        if data.get("version") == VERSION:
            data["summary"] = summarize(data["progress"], data["shares"])
            return data
    config = json.loads((run_dir / "run_config.json").read_text())
    mask = load_mask(garment_mask_path(resolve_input(config["inputs"]["person"])), ANALYSIS_SIZE)
    window = gaussian_filter(mask.astype(np.float64), 3)  # soft edge: no artificial fine-band energy at the mask border
    labs = [rgb2lab(load_rgb(p)) for p in sorted((run_dir / "frames_x0").glob("step_*.png"))]
    final_bands = split_bands(labs[-1])

    shares = {k: [float("nan")] for k in BANDS + CHANNELS}
    change_rms = [float("nan")]
    dist = {k: [] for k in BANDS + CHANNELS}
    prev_bands = None
    for i, lab in enumerate(labs):
        bands = split_bands(lab)
        for b in BANDS:
            dist[b].append(np.sqrt(energy(bands[b] - final_bands[b], window)))
        diff_final = lab - labs[-1]
        dist["L"].append(np.sqrt(energy(diff_final, window, slice(0, 1))))
        dist["chroma"].append(np.sqrt(energy(diff_final, window, slice(1, 3))))
        if prev_bands is not None:
            e = {b: energy(bands[b] - prev_bands[b], window) for b in BANDS}
            tot = sum(e.values())
            for b in BANDS:
                shares[b].append(e[b] / tot if tot > 0 else float("nan"))
            d = lab - labs[i - 1]
            eL, eC = energy(d, window, slice(0, 1)), energy(d, window, slice(1, 3))
            shares["L"].append(eL / (eL + eC) if eL + eC > 0 else float("nan"))
            shares["chroma"].append(eC / (eL + eC) if eL + eC > 0 else float("nan"))
            change_rms.append(float(np.sqrt((eL + eC) / window.sum())))
        prev_bands = bands

    progress = {}
    for k, d in dist.items():
        d = np.asarray(d)
        base = d[START]
        p = np.clip(1.0 - d / base, 0.0, 1.0) if base > 0 else np.ones_like(d)
        p[:START] = np.nan
        progress[k] = p.tolist()
    data = {
        "version": VERSION,
        "progress": progress,
        "shares": shares,
        "change_rms": change_rms,
        "summary": summarize(progress, shares),
    }
    cache.write_text(json.dumps(data))
    return data


def nanmean_curves(curves):
    arr = np.array(curves, dtype=np.float64)
    with warnings.catch_warnings():  # every curve is NaN at the undefined first step(s)
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[1]).parse_args()
    categories_of = load_categories()
    runs = sorted(d for prefix in MODELS for d in (REPO_ROOT / "outputs").glob(f"{prefix}_*")
                  if (d / "run_config.json").exists())
    rows, data = [], {}
    for i, d in enumerate(runs):
        prefix, pair = parse_run(d)
        name = MODELS[prefix][0]
        data[(name, pair)] = analyze_run(d)
        rows.append({"model": name, "pair": pair, "category": categories_of.get(pair, "uncategorized"),
                     **data[(name, pair)]["summary"]})
        print(f"[decompose] {i + 1}/{len(runs)} {d.name}", end="\r", flush=True)
    print()
    names = [v[0] for v in MODELS.values() if any(r["model"] == v[0] for r in rows)]
    cols = [c for c in rows[0] if c not in ("model", "pair", "category")]

    out = REPO_ROOT / "outputs" / "_analysis" / "change_decomposition"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "pair", "model"] + cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    for fname, keys in (("by_model.csv", ("model",)), ("by_category.csv", ("category", "model"))):
        groups = {}
        for r in rows:
            groups.setdefault(tuple(r[k] for k in keys), []).append(r)
        with open(out / fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(keys) + ["n"] + [f"{c}_{s}" for c in cols for s in ("mean", "std")])
            for key, g in groups.items():
                vals = []
                for c in cols:
                    v = [r[c] for r in g]
                    vals += [f"{statistics.fmean(v):.4f}", f"{statistics.stdev(v) if len(v) > 1 else 0:.4f}"]
                w.writerow(list(key) + [len(g)] + vals)

    # statistics: is each index different from 0 within a model; do models differ (paired by pair)
    lines = []
    meaning = {"coarse_to_fine_index": "> 0: coarse scales resolved before fine",
               "luma_before_chroma_index": "> 0: luminance resolved before chroma",
               "detail_phase_headstart": "higher: detail phase starts earlier; step = (1 - value) * 49"}
    for idx, what in meaning.items():
        lines.append(f"== {idx}  ({what})")
        by = {n: {r["pair"]: r[idx] for r in rows if r["model"] == n} for n in names}
        for n in names:
            v = np.array(list(by[n].values()))
            if idx.endswith("_index"):
                p = wilcoxon(v).pvalue if np.any(v != 0) else 1.0
                lines.append(f"  {n:14} median {np.median(v):+.3f}, > 0 in {np.mean(v > 0):4.0%} of {len(v)} pairs, Wilcoxon vs 0 p={p:.1e}")
            else:
                lines.append(f"  {n:14} median {np.median(v):.3f} (step {(1 - np.median(v)) * 49:.0f}), IQR {np.percentile(v, 25):.2f}-{np.percentile(v, 75):.2f}")
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                common = sorted(set(by[names[a]]) & set(by[names[b]]))
                x = np.array([by[names[a]][p] for p in common]); y = np.array([by[names[b]][p] for p in common])
                p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
                lines.append(f"  {names[a]} vs {names[b]}: median diff {np.median(x - y):+.3f}, paired Wilcoxon p={p:.1e}")
    (out / "stats.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # progress per band, one panel per model (+ L/chroma row)
    fig, axes = plt.subplots(2, len(names), figsize=(5.5 * len(names), 8), squeeze=False)
    for j, n in enumerate(names):
        runs_n = [v for (m, _), v in data.items() if m == n]
        for row, keys in enumerate((BANDS, CHANNELS)):
            ax = axes[row, j]
            for k in keys:
                mean, std = nanmean_curves([r["progress"][k] for r in runs_n])
                x = np.arange(len(mean))
                ax.plot(x, mean, color=BAND_COLORS[k], label=k)
                ax.fill_between(x, mean - std, mean + std, color=BAND_COLORS[k], alpha=0.12, linewidth=0)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{n} (n={len(runs_n)}) -- {'scale bands' if row == 0 else 'luminance vs chroma'}", fontsize=10)
            ax.set_xlabel("denoising step")
            ax.set_ylabel("progress to final (own start = 0)")
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Band progress: share of each band's distance to the final image already removed (higher = further along)")
    fig.tight_layout()
    fig.savefig(out / "progress_by_band.png", dpi=110)
    plt.close(fig)

    # where does the step-to-step change happen: stacked shares + change magnitude
    fig, axes = plt.subplots(2, len(names), figsize=(5.5 * len(names), 7), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1]})
    for j, n in enumerate(names):
        runs_n = [v for (m, _), v in data.items() if m == n]
        means = [nanmean_curves([r["shares"][b] for r in runs_n])[0] for b in BANDS]
        x = np.arange(len(means[0]))
        ax = axes[0, j]
        ax.stackplot(x[1:], *[m[1:] for m in means], labels=BANDS, colors=[BAND_COLORS[b] for b in BANDS], alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_title(f"{n}: share of each step's change per band", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        rms, _ = nanmean_curves([r["change_rms"] for r in runs_n])
        axes[1, j].plot(x[1:], rms[1:], color="black")
        axes[1, j].set_yscale("log")
        axes[1, j].set_ylabel("change size (Lab RMS)", fontsize=8)
        axes[1, j].set_xlabel("denoising step")
    fig.suptitle("What is being edited when (shares of step-to-step change; not a score)")
    fig.tight_layout()
    fig.savefig(out / "change_shares.png", dpi=110)
    plt.close(fig)

    # indices per category and model
    cats = list(dict.fromkeys(categories_of.values()))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    width = 0.8 / len(names)
    for ax, idx, title in zip(axes, ("coarse_to_fine_index", "luma_before_chroma_index"),
                              ("coarse-to-fine index (> 0: coarse scales resolved first)",
                               "luminance-before-chroma index (> 0: brightness resolved first)")):
        for i, n in enumerate(names):
            m, s = [], []
            for c in cats + ["ALL"]:
                v = [r[idx] for r in rows if r["model"] == n and (c == "ALL" or r["category"] == c)]
                m.append(np.mean(v) if v else np.nan)
                s.append(np.std(v) if v else 0)
            x = np.arange(len(cats) + 1) + (i - (len(names) - 1) / 2) * width
            ax.bar(x, m, width, yerr=s, color=dict(MODELS.values())[n], label=n, capsize=2, error_kw={"linewidth": 0.8})
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(cats) + 1))
        ax.set_xticklabels(cats + ["ALL"], rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "index_by_category.png", dpi=110)
    plt.close(fig)
    print(f"Wrote {out}/")


if __name__ == "__main__":
    main()
