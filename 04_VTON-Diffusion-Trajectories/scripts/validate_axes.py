"""
Validate that each trajectory axis in metrics.py measures the property it is
named after -- and not the others.

Takes the final try-on image of every run, applies controlled perturbations
that change ONE property at known strengths, and scores the perturbed image
against the unperturbed one with exactly the code the trajectory analysis
uses (metrics.scores_vs_reference, same garment mask and resolution):

  blur      Gaussian blur (sigma in px at 384x512)  -> removes fine detail
  color     constant Lab (a, b) offset of Delta-E d -> changes color only
  warp      smooth random displacement (max px)     -> moves structure,
                                                       keeps colors/detail
  noise     Gaussian pixel noise (std, 0-1 scale)   -> robustness check: no
                                                       axis should read noise
                                                       as color/structure/detail

A well-behaved axis drops strongly for "its" perturbation and little for the
others. Output is a specificity table: the DROP (1 - score) per axis and
perturbation, averaged over images -- here, unlike everywhere else in this
project, HIGHER means the axis REACTS MORE (that is the quantity being
validated, not a quality).

Writes outputs/_analysis/axis_validation/{specificity.csv, specificity.png}.

Usage:
    python scripts/validate_axes.py                 # all runs' final images
    python scripts/validate_axes.py --max-images 30
"""
import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.color import lab2rgb, rgb2lab

from metrics import ANALYSIS_SIZE, Reference, garment_mask_path, load_mask, load_rgb, resolve_input, scores_vs_reference

REPO_ROOT = Path(__file__).resolve().parent.parent
AXES = ("color", "structure", "texture", "pattern")


def blur(rgb, sigma, rng):
    return np.clip(np.stack([gaussian_filter(rgb[..., c], sigma) for c in range(3)], -1), 0, 1)


def color_shift(rgb, delta_e, rng):
    lab = rgb2lab(rgb)
    angle = rng.uniform(0, 2 * np.pi)  # random hue direction, fixed magnitude
    lab[..., 1] += delta_e * np.cos(angle)
    lab[..., 2] += delta_e * np.sin(angle)
    return np.clip(lab2rgb(lab), 0, 1).astype(np.float32)


def warp(rgb, max_px, rng):
    h, w = rgb.shape[:2]
    fields = []
    for _ in range(2):
        f = gaussian_filter(rng.standard_normal((h, w)), 25)  # smooth: moves regions, not pixels
        fields.append(f / (np.abs(f).max() + 1e-8) * max_px)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    coords = [yy + fields[0], xx + fields[1]]
    # cubic: linear interpolation visibly softens the image, which would make
    # texture look warp-sensitive when it's really reacting to the resampling
    return np.clip(np.stack([map_coordinates(rgb[..., c], coords, order=3, mode="reflect") for c in range(3)], -1), 0, 1)


def noise(rgb, std, rng):
    return np.clip(rgb + rng.normal(0, std, rgb.shape).astype(np.float32), 0, 1)


PERTURBATIONS = {  # name: (function, strengths, unit, the axis it is meant to hit)
    "blur": (blur, (1, 2, 4), "sigma px", "texture"),
    "color": (color_shift, (4, 8, 16), "Delta-E", "color"),
    "warp": (warp, (2, 4, 8), "max px", "structure"),
    "noise": (noise, (0.02, 0.05, 0.1), "std", "none"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-images", type=int, default=None, help="random subset of runs (default: all)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    runs = sorted(d for d in (REPO_ROOT / "outputs").glob("*_*") if (d / "run_config.json").exists())
    if args.max_images:
        runs = random.Random(args.seed).sample(runs, min(args.max_images, len(runs)))
    rng = np.random.default_rng(args.seed)

    drops = {(name, s): {a: [] for a in AXES} for name, (_, strengths, _, _) in PERTURBATIONS.items() for s in strengths}
    for i, run in enumerate(runs):
        config = json.loads((run / "run_config.json").read_text())
        mask = load_mask(garment_mask_path(resolve_input(config["inputs"]["person"])), ANALYSIS_SIZE)
        final = load_rgb(run / "final.png")
        ref = Reference(final, mask)
        for name, (fn, strengths, _, _) in PERTURBATIONS.items():
            for s in strengths:
                scores = scores_vs_reference(fn(final, s, rng).astype(np.float32), ref)
                for a in AXES:
                    drops[(name, s)][a].append(1.0 - scores[a])
        print(f"[validate] {i + 1}/{len(runs)} {run.name}", end="\r", flush=True)
    print()

    out = REPO_ROOT / "outputs" / "_analysis" / "axis_validation"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out / "specificity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["perturbation", "strength", "unit", "target_axis", "n"] + [f"{a}_drop_mean" for a in AXES] + [f"{a}_drop_std" for a in AXES])
        for (name, s), d in drops.items():
            _, _, unit, target = PERTURBATIONS[name]
            means = [float(np.mean(d[a])) for a in AXES]
            rows.append((f"{name} {s:g}", target, means))
            w.writerow([name, s, unit, target, len(d[AXES[0]])] + [f"{m:.4f}" for m in means] + [f"{np.std(d[a]):.4f}" for a in AXES])

    mat = np.array([r[2] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1.5))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=max(0.5, float(mat.max())), aspect="auto")
    ax.set_xticks(range(len(AXES)))
    ax.set_xticklabels(AXES)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r[0]}  (-> {r[1]})" for r in rows], fontsize=8)
    for i in range(len(rows)):
        for j, a in enumerate(AXES):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if mat[i, j] > 0.35 else "white", fontweight="bold" if a == rows[i][1] else "normal")
    fig.colorbar(im, label="drop (1 - score): higher = axis reacts more")
    ax.set_title(f"Axis specificity, {len(runs)} final images\n(bold = the axis the perturbation targets)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "specificity.png", dpi=120)
    print(f"Wrote {out}/specificity.csv and .png")


if __name__ == "__main__":
    main()
