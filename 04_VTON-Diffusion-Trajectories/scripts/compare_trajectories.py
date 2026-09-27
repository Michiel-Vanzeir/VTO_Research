"""
Cross-model comparison of the per-run trajectory metrics (see metrics.py for
what each axis measures). ALL NUMBERS ARE IN [0, 1] AND HIGHER IS BETTER:
higher auc / headstart = the model commits to its final result earlier,
higher print_fidelity_final = the final try-on reproduces the garment's
pattern better.

Runs are grouped by pair and by garment category (from cluster/pairs.txt,
written by select_pairs.py), so every statement is backed by several
garments per category rather than one.

Writes to outputs/_comparison/ (regenerated from scratch on every run):
  runs.csv                one row per run: every summary score
  by_category.csv         mean / std / n per (category, model)
  by_model.csv            mean / std / n per model over all pairs
  summary_auc.png         per axis: AUC per category and model (+ std error bars)
  summary_headstart.png   the same for headstart
  category_<cat>.png      mean curve per model (+/- std band) over that category
  category_all.png        the same over all pairs
  pair_<name>.png         every model's curves for one pair

Usage:
    python compare_trajectories.py                        # all runs in outputs/
    python compare_trajectories.py outputs/idm_* outputs/ootd_*
"""
import argparse
import csv
import statistics
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import AXES, SETTLE_THRESHOLD, load_or_compute

REPO_ROOT = Path(__file__).resolve().parent.parent
PAIRS_FILE = REPO_ROOT / "cluster" / "pairs.txt"

# run-dir prefix -> (display name, plot color); order = order in tables/plots
MODELS = {"catvton": ("CatVTON", "#2563eb"), "ootd": ("OOTDiffusion", "#ea580c"), "idm": ("IDM-VTON", "#16a34a")}


def load_categories() -> dict:
    """pair name -> category, in pairs.txt order."""
    cats = {}
    if PAIRS_FILE.exists():
        for line in PAIRS_FILE.read_text().split("\n"):
            parts = line.split()
            if len(parts) == 4 and not line.startswith("#"):
                cats[parts[0]] = parts[1]
    return cats


def parse_run(run_dir: Path):
    """outputs/<model>_<pair> -> (model prefix, pair name), or None."""
    for prefix in MODELS:
        if run_dir.name.startswith(prefix + "_"):
            return prefix, run_dir.name[len(prefix) + 1:]
    return None


def score_columns():
    cols = []
    for key, _, _, has_headstart in AXES:
        cols.append(f"{key}_auc")
        if has_headstart:
            cols.append(f"{key}_headstart")
    cols.append("print_fidelity_final")
    return cols


def run_scores(result: dict) -> dict:
    s = result["summary"]
    row = {}
    for key, _, _, has_headstart in AXES:
        row[f"{key}_auc"] = s[key]["auc"]
        if has_headstart:
            row[f"{key}_headstart"] = s[key]["headstart"]
    row["print_fidelity_final"] = s["print_fidelity"]["final"]
    return row


def mean_std(values):
    v = [x for x in values if x is not None]
    if not v:
        return None, None, 0
    return statistics.fmean(v), (statistics.stdev(v) if len(v) > 1 else 0.0), len(v)


def write_grouped_csv(path: Path, rows: list, group_keys: tuple, order: dict):
    cols = score_columns()
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(group_keys) + ["n"] + [f"{c}_{s}" for c in cols for s in ("mean", "std")])
        for key in sorted(groups, key=lambda k: tuple(order.get(x, 99) for x in k)):
            g = groups[key]
            out = list(key) + [len(g)]
            for c in cols:
                m, s, _ = mean_std([r[c] for r in g])
                out += ["" if m is None else f"{m:.4f}", "" if s is None else f"{s:.4f}"]
            w.writerow(out)


def plot_curves(title: str, results_by_model: dict, out_path: Path):
    """results_by_model: display name -> list of metric results. One line per
    model (mean over its results), shaded +/- std when there are several."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (key, axis_title, ylabel, has_headstart) in zip(axes.flat, AXES):
        for name, color in MODELS.values():
            results = results_by_model.get(name)
            if not results:
                continue
            arr = np.array([r["curves"][key] for r in results], dtype=np.float64)
            with warnings.catch_warnings():  # stability is NaN at step 0 for every run
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(arr, axis=0)
            x = np.arange(arr.shape[1])
            ax.plot(x, mean, color=color, label=f"{name} (n={len(results)})")
            if len(results) > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    std = np.nanstd(arr, axis=0)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        if has_headstart:
            ax.axhline(SETTLE_THRESHOLD, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(axis_title, fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("denoising step")
        ax.set_ylabel(ylabel, fontsize=8)
    axes.flat[0].legend(fontsize=8, loc="lower right")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, 0.5, "all scores: higher = better / closer\n"
                       "line = mean, band = +/- 1 std over pairs\n"
                       f"dotted line: settle threshold {SETTLE_THRESHOLD}", fontsize=9, va="center")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_summary(rows: list, categories: list, metric_cols: list, titles: list, out_path: Path, suptitle: str):
    ncols = 4
    nrows = -(-len(metric_cols) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
    groups = categories + ["ALL"]
    width = 0.8 / len(MODELS)
    for ax, col, title in zip(axes.flat, metric_cols, titles):
        for i, (name, color) in enumerate(MODELS.values()):
            means, stds = [], []
            for g in groups:
                m, s, _ = mean_std([r[col] for r in rows if r["model"] == name and (g == "ALL" or r["category"] == g)])
                means.append(np.nan if m is None else m)
                stds.append(0 if s is None else s)
            x = np.arange(len(groups)) + (i - (len(MODELS) - 1) / 2) * width
            ax.bar(x, means, width, yerr=stds, color=color, label=name, capsize=2, error_kw={"linewidth": 0.8})
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
        ax.axvline(len(groups) - 1.5, color="gray", linewidth=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
    for ax in axes.flat[len(metric_cols):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="*", help="default: every outputs/<model>_* run")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "_comparison"))
    args = p.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs] or sorted(
        d for prefix in MODELS for d in (REPO_ROOT / "outputs").glob(f"{prefix}_*"))
    categories_of = load_categories()

    rows, results = [], {}  # results[(model display name, pair)] = metrics result
    for d in run_dirs:
        parsed = parse_run(d)
        if parsed is None:
            print(f"[skip] {d}: not an outputs/<catvton|ootd|idm>_<pair> run dir")
            continue
        if not (d / "run_config.json").exists():
            print(f"[skip] {d}: no run_config.json (run incomplete or failed)")
            continue
        prefix, pair = parsed
        name = MODELS[prefix][0]
        result = load_or_compute(d)
        results[(name, pair)] = result
        rows.append({"model": name, "pair": pair, "category": categories_of.get(pair, "uncategorized"),
                     **run_scores(result)})
    if not rows:
        raise SystemExit("no completed runs to compare")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in list(out.glob("*.png")) + list(out.glob("*.csv")):  # generated dir: no leftovers from older versions
        stale.unlink()

    # table/plot order: categories as in pairs.txt, models as in MODELS
    cat_order = list(dict.fromkeys(categories_of.values())) + ["uncategorized"]
    categories = [c for c in cat_order if any(r["category"] == c for r in rows)]
    order = {**{c: i for i, c in enumerate(cat_order)}, **{v[0]: i for i, v in enumerate(MODELS.values())}}
    rows.sort(key=lambda r: (order[r["category"]], r["pair"], order[r["model"]]))

    cols = score_columns()
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "pair", "model"] + cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    write_grouped_csv(out / "by_category.csv", rows, ("category", "model"), order)
    write_grouped_csv(out / "by_model.csv", rows, ("model",), order)

    auc_cols = [f"{k}_auc" for k, *_ in AXES] + ["print_fidelity_final"]
    auc_titles = [f"{t} -- auc" for _, t, *_ in AXES] + ["PRINT FIDELITY -- final image"]
    plot_summary(rows, categories, auc_cols, auc_titles, out / "summary_auc.png",
                 "AUC per category (higher = closer to its final result earlier; error bar = std over pairs)")
    hs_cols = [f"{k}_headstart" for k, _, _, hs in AXES if hs]
    hs_titles = [f"{t} -- headstart" for _, t, _, hs in AXES if hs]
    plot_summary(rows, categories, hs_cols, hs_titles, out / "summary_headstart.png",
                 f"Headstart per category (share of steps left when the curve settles >= {SETTLE_THRESHOLD}; higher = earlier)")

    for cat in categories + ["all"]:
        by_model = {}
        for (name, pair), res in results.items():
            if cat == "all" or categories_of.get(pair, "uncategorized") == cat:
                by_model.setdefault(name, []).append(res)
        plot_curves(f"category: {cat}", by_model, out / f"category_{cat}.png")
    for pair in sorted({pair for _, pair in results}):
        by_model = {name: [res] for (name, p), res in results.items() if p == pair}
        if len(by_model) > 1:
            plot_curves(f"pair: {pair} ({categories_of.get(pair, 'uncategorized')})", by_model, out / f"pair_{pair}.png")

    print(f"\n{len(rows)} runs, {len(categories)} categories. Mean over all pairs (higher = better):")
    print(f"{'':14}" + "".join(f"{c.replace('_auc', ''):>22}" for c in auc_cols))
    for name in (v[0] for v in MODELS.values()):
        vals = [mean_std([r[c] for r in rows if r["model"] == name])[0] for c in auc_cols]
        if any(v is not None for v in vals):
            print(f"{name:14}" + "".join(f"{v:22.3f}" if v is not None else f"{'-':>22}" for v in vals))
    print(f"\nWrote {out}/: runs.csv, by_category.csv, by_model.csv, summary_*.png, category_*.png, pair_*.png")


if __name__ == "__main__":
    main()
