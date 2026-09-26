"""
Cross-run / cross-model comparison of diffusion-trajectory metrics
(see metrics.py for what structure_onset/texture_onset mean).

Groups the given run directories by "pair" (the run-dir name with a
leading `<model>_` prefix stripped, e.g. `ootd_pair1_solid` and
`pair1_solid` both belong to pair `pair1_solid`) and, for every pair that
has runs from more than one model, plots their pose/color/texture/print
curves together (see metrics.py's module docstring for what each of those
four axes means) and tabulates the onset gap between models on each axis.
This is the script that actually answers "does model A settle pose,
color, texture, or the specific print pattern earlier or later than model
B", across the different garment types tested.

Usage:
    python compare_trajectories.py outputs/catvton_pair*_* outputs/ootd_pair*_* outputs/idm_pair*_* --crop-to-mask
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import compute_run_metrics

KNOWN_PREFIXES = ("ootd_", "catvton_", "idm_")


def pair_key(run_dir: Path) -> str:
    name = run_dir.name
    for prefix in KNOWN_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def load_or_compute(run_dir: Path, crop_to_mask: bool) -> dict:
    suffix = "_masked" if crop_to_mask else ""
    cached = run_dir / f"metrics{suffix}.json"
    if cached.exists():
        result = json.loads(cached.read_text())
        if "pose_pck_to_final" in result["metrics"]:  # older caches predate the OpenPose POSE axis
            return result
    result = compute_run_metrics(run_dir, crop_to_mask=crop_to_mask)
    cached.write_text(json.dumps(result, indent=2))
    return result


# (subplot title, onset key, metric key to plot, y-label)
AXIS_SPECS = (
    ("POSE", "pose_onset", "pose_pck_to_final", "OpenPose PCK vs final skeleton"),
    ("COLOR", "color_onset", "color_dist_to_final", "Lab distance to final"),
    ("TEXTURE", "texture_onset", "high_freq_frac", "high-freq FFT fraction"),
    ("PRINT", "print_onset", "print_pattern_dist", "spectral distance to cloth ref"),
    ("STABILITY", "stability_onset", "step_delta", "RMS change vs previous step"),
    ("POSE REFINE", "pose_refine_onset", "pose_kp_err_to_final", "keypoint error vs final (torso lengths)"),
)


def plot_pair_comparison(pair: str, results: list, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = plt.cm.tab10.colors

    for ax, (title, onset_key, metric_key, ylabel) in zip(axes.flat, AXIS_SPECS):
        for i, r in enumerate(results):
            color = colors[i % len(colors)]
            steps = range(len(r["metrics"][metric_key]))
            ax.plot(steps, r["metrics"][metric_key], color=color, label=r["model"])
            onset = r["onsets"][onset_key]
            if onset is not None:
                ax.axvline(onset["step"], color=color, linestyle=":", alpha=0.7)
        if title == "POSE":
            ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"{pair} -- {title}")
        ax.set_xlabel("denoising step")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(AXIS_SPECS):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--crop-to-mask", action="store_true")
    p.add_argument("--out-dir", default=None, help="defaults to outputs/_comparison next to the first run dir's outputs/")
    args = p.parse_args()

    # Skip runs that never finished (e.g. a failed cluster job leaves an
    # output dir behind but no run_config.json).
    run_dirs = []
    for d in map(Path, args.run_dirs):
        if (d / "run_config.json").exists():
            run_dirs.append(d)
        else:
            print(f"[skip] {d}: no run_config.json (run incomplete or failed)")
    if not run_dirs:
        raise SystemExit("no completed runs to compare")
    out_root = Path(args.out_dir) if args.out_dir else run_dirs[0].parent / "_comparison"
    out_root.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list] = {}
    for run_dir in run_dirs:
        result = load_or_compute(run_dir, args.crop_to_mask)
        groups.setdefault(pair_key(run_dir), []).append(result)

    summary_rows = []
    for pair, results in sorted(groups.items()):
        if len(results) < 2:
            print(f"[skip] {pair}: only one model run found ({results[0]['model']}), nothing to compare")
            continue
        suffix = "_masked" if args.crop_to_mask else ""
        plot_pair_comparison(pair, results, out_root / f"{pair}{suffix}.png")
        for r in results:
            row = {"pair": pair, "model": r["model"]}
            for key, onset in r["onsets"].items():
                row[f"{key}_step"] = onset["step"] if onset else ""
                row[f"{key}_t"] = onset["timestep"] if onset else ""
            row["pose_saturated"] = bool(r["onsets"]["pose_onset"] and r["onsets"]["pose_onset"].get("saturated"))
            summary_rows.append(row)
        print(f"[ok] {pair}: compared {[r['model'] for r in results]} -> {out_root / f'{pair}{suffix}.png'}")

    if summary_rows:
        suffix = "_masked" if args.crop_to_mask else ""
        csv_path = out_root / f"onset_summary{suffix}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
