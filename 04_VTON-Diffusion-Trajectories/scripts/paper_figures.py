"""
Paper-sized vector figures for paper2/ (VISAPP, SCITEPRESS two-column).
Only the six axes used in the paper (pose, color, structure, texture,
pattern, stability); print fidelity and the fork experiment are left out.

Reads the cached per-run results (metrics.json, change_decomposition.json)
and outputs/_analysis/axis_validation/specificity.csv, so run metrics.py /
compare_trajectories.py, change_decomposition.py and validate_axes.py first.

Writes current_paper/figures/{curves_all,curves_by_category,summary_auc,axis_specificity,band_progress,x0_strip*}.pdf

Usage:
    python scripts/paper_figures.py
"""
import csv
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_trajectories import MODELS, load_categories

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "current_paper" / "figures"
AXES = (("pose", "Pose"), ("color", "Colour"), ("structure", "Structure"),
        ("texture", "Texture"), ("pattern", "Pattern"), ("stability", "Stability"))
GARMENT_AXES = AXES[1:5]
TEXT_W, COL_W = 7.0, 3.35  # inches
# Fixed categorical order (one hue per category, never cycled); uniform
# designs are drawn dashed, repeating patterns solid, so identity never rests
# on colour alone.
CATEGORY_STYLE = {"solid": ("#2a78d6", "--"), "lettering": ("#eb6834", "--"), "graphic": ("#1baf7a", "--"),
                  "structural": ("#eda100", "--"), "stripe": ("#e87ba4", "-"), "floral": ("#008300", "-"),
                  "dot": ("#6250d6", "-"), "check": ("#e34948", "-")}

# x0-hat examples next to the stripe strip: a floral print (IDM-VTON as late
# as CatVTON, unlike stripes) and lettering (all models early). The appendix
# holds the most typical pair (closest to the category mean at step 10) of
# each remaining category, plus floral_02, where OOTDiffusion is late too.
STRIP_STEPS = (1, 5, 10, 20, 30, 49)
MAIN_STRIPS = {"pair4_print": "x0_strip_floral.pdf", "lettering_03": "x0_strip_lettering.pdf"}
APPENDIX_PAIRS = ("solid_02", "graphic_02", "structural_04", "dot_04", "check_02", "floral_02")
# Appendix grid: three pairs side by side, five steps each, to fit in half a page.
APPENDIX_STEPS = (1, 5, 10, 20, 49)

plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42, "axes.linewidth": 0.6})


def nanmean_std(curves):
    arr = np.array(curves, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(arr, 0), np.nanstd(arr, 0)


def load_runs(fname):
    cats = load_categories()
    data = {}
    for prefix, (name, _) in MODELS.items():
        for pair in cats:
            path = REPO_ROOT / "outputs" / f"{prefix}_{pair}" / fname
            if path.exists():
                data.setdefault(name, []).append((cats[pair], json.loads(path.read_text())))
    return data


def curves_figure(runs):
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_W, 3.6), sharex=True)
    for ax, (key, title) in zip(axes.flat, AXES):
        for name, color in MODELS.values():
            mean, std = nanmean_std([r["curves"][key] for _, r in runs[name]])
            x = np.arange(len(mean))
            ax.plot(x, mean, color=color, linewidth=1.1, label=name)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, linewidth=0.4)
    for ax in axes[1]:
        ax.set_xlabel("denoising step")
    for ax in axes[:, 0]:
        ax.set_ylabel("score (1 = final)")
    axes[1, 2].set_ylabel("score (1 = unchanged)")
    axes[0, 0].legend(loc="lower right", frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "curves_all.pdf")
    plt.close(fig)


def auc_figure(runs):
    """Mean AUC per category and model as points with +/- 1 std whiskers
    (points, not bars: the axis does not start at zero)."""
    cats = list(CATEGORY_STYLE)
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_W, 2.3), sharey=True)
    offset = 0.22
    for ax, (key, title) in zip(axes, GARMENT_AXES):
        for i, (name, color) in enumerate(MODELS.values()):
            m, s = [], []
            for c in cats:
                v = [r["summary"][key]["auc"] for cat, r in runs[name] if cat == c]
                m.append(np.mean(v))
                s.append(np.std(v))
            ax.errorbar(np.arange(len(cats)) + (i - 1) * offset, m, yerr=s, fmt="o", color=color, label=name,
                        markersize=2.8, elinewidth=0.6, capsize=0)
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels(cats, rotation=60, ha="right")
        ax.set_xlim(-0.6, len(cats) - 0.4)
        ax.set_ylim(0.4, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    axes[0].set_ylabel("trajectory AUC")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "summary_auc.pdf")
    plt.close(fig)


def category_curves_figure(runs):
    """Within-model view: one panel per (axis, model), one mean curve per
    garment category, so the shape of the trajectory (early jump and plateau
    vs. steady rise) can be compared between categories of the same model."""
    cats = list(CATEGORY_STYLE)
    names = [n for n, _ in MODELS.values()]
    fig, axes = plt.subplots(len(GARMENT_AXES), len(names), figsize=(TEXT_W, 6.0), sharex=True, sharey=True)
    for row, (key, title) in enumerate(GARMENT_AXES):
        for col, name in enumerate(names):
            ax = axes[row, col]
            for c in cats:
                mean, _ = nanmean_std([r["curves"][key] for cat, r in runs[name] if cat == c])
                color, ls = CATEGORY_STYLE[c]
                ax.plot(np.arange(len(mean)), mean, color=color, linestyle=ls, linewidth=1.0, label=c)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25, linewidth=0.4)
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(title)
            if row == len(GARMENT_AXES) - 1:
                ax.set_xlabel("denoising step")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=8, frameon=False,
               handlelength=2.6, columnspacing=1.2)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.965))
    fig.savefig(OUT / "curves_by_category.pdf")
    plt.close(fig)


def specificity_figure():
    rows = list(csv.DictReader(open(REPO_ROOT / "outputs" / "_analysis" / "axis_validation" / "specificity.csv")))
    keys = [k for k, _ in GARMENT_AXES]
    names = {"blur": "blur $\\sigma$", "color": "colour $\\Delta E$", "warp": "warp px (max)", "noise": "noise std"}
    mat = np.array([[float(r[f"{k}_drop_mean"]) for k in keys] for r in rows])
    fig, ax = plt.subplots(figsize=(COL_W, 3.0))
    ax.imshow(mat, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([t for _, t in GARMENT_AXES])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{names[r['perturbation']]} {float(r['strength']):g}" for r in rows])
    target = {"blur": "texture", "color": "color", "warp": "structure"}
    for i, r in enumerate(rows):
        for j, k in enumerate(keys):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if mat[i, j] > 0.5 else "black",
                    fontweight="bold" if target.get(r["perturbation"]) == k else "normal")
    for y in (2.5, 5.5, 8.5):
        ax.axhline(y, color="black", linewidth=0.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "axis_specificity.pdf")
    plt.close(fig)


def band_figure(runs):
    colors = {"coarse": "#1d4ed8", "mid": "#7c3aed", "fine": "#db2777"}
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 1.9), sharey=True)
    for ax, (name, _) in zip(axes, MODELS.values()):
        for band, color in colors.items():
            mean, std = nanmean_std([r["progress"][band] for _, r in runs[name]])
            x = np.arange(len(mean))
            ax.plot(x, mean, color=color, linewidth=1.1, label=band)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        ax.set_title(name)
        ax.set_xlabel("denoising step")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, linewidth=0.4)
    axes[0].set_ylabel("band progress")
    axes[0].legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "band_progress.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs("metrics.json")
    curves_figure(runs)
    auc_figure(runs)
    category_curves_figure(runs)
    specificity_figure()
    band_figure(load_runs("change_decomposition.json"))
    strip_figure()
    for pair, out in MAIN_STRIPS.items():
        strip_figure(pair, out)
    appendix_strip_figure()
    print(f"Wrote {OUT}")


def _x0(prefix, pair, step):
    from PIL import Image
    img = Image.open(REPO_ROOT / "outputs" / f"{prefix}_{pair}" / "frames_x0" / f"step_{step:03d}.png").convert("RGB")
    return img.resize((192, 256), Image.LANCZOS)


def strip_figure(pair="stripe_01", out="x0_strip.pdf", steps=STRIP_STEPS):
    """Decoded x0-hat at a few steps (columns) for every model (rows), one pair."""
    fig, axes = plt.subplots(len(MODELS), len(steps), figsize=(COL_W, 2.42))
    for r, (prefix, (name, _)) in enumerate(MODELS.items()):
        for c, s in enumerate(steps):
            ax = axes[r, c]
            ax.imshow(_x0(prefix, pair, s))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"step {s}", fontsize=6.5)
            if c == 0:
                ax.set_ylabel(name, fontsize=6.5)
    fig.subplots_adjust(left=0.06, right=1, top=0.94, bottom=0, wspace=0.03, hspace=0.03)
    fig.savefig(OUT / out, dpi=300)
    plt.close(fig)


def appendix_strip_figure(pairs=APPENDIX_PAIRS, steps=APPENDIX_STEPS, per_row=3):
    """Grid of x0-hat strips, `per_row` pairs side by side per block row."""
    cats = load_categories()
    cloth = {ln.split()[0]: ln.split()[3].removesuffix("_00")
             for ln in (REPO_ROOT / "cluster" / "pairs.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")}
    n_rows = -(-len(pairs) // per_row)
    img_w = (TEXT_W - 0.25) / (per_row * len(steps))
    fig = plt.figure(figsize=(TEXT_W, n_rows * (len(MODELS) * img_w * 4 / 3 + 0.42)))
    outer = fig.add_gridspec(n_rows, per_row, left=0.025, right=1, top=0.94, bottom=0, wspace=0.05, hspace=0.22)
    for i, pair in enumerate(pairs):
        inner = outer[i // per_row, i % per_row].subgridspec(len(MODELS), len(steps), wspace=0.03, hspace=0.03)
        for r, (prefix, (name, _)) in enumerate(MODELS.items()):
            for c, s in enumerate(steps):
                ax = fig.add_subplot(inner[r, c])
                ax.imshow(_x0(prefix, pair, s))
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"step {s}", fontsize=6, pad=2)
                if c == 0 and i % per_row == 0:
                    ax.set_ylabel(name, fontsize=6)
                if r == 0 and c == 0:
                    ax.text(0, 1.27, f"({chr(97 + i)}) {cats[pair]} (garment {cloth[pair]})",
                            transform=ax.transAxes, fontsize=7, fontweight="bold")
    fig.savefig(OUT / "x0_strip_appendix.pdf", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
