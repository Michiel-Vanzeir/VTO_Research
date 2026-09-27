"""
Per-step analysis of one captured diffusion trajectory (frames_x0/ written by
catvton_trajectory.py, ootd_trajectory.py or idmvton_trajectory.py).

EVERY SCORE IS IN [0, 1] AND HIGHER IS ALWAYS BETTER: for per-step curves
"higher" means "closer to where this run ends up" (or, for print_fidelity,
closer to the source garment photo); for the per-run summaries it means
"committed earlier" / "better result". Nothing needs to be mentally
inverted when reading plots or tables.

To keep the three models comparable, every frame is first resampled to one
analysis resolution (ANALYSIS_SIZE; CatVTON's native 384x512), and all
garment-region scores use the SAME pixel mask for every model: the VITON-HD
dataset's `agnostic-mask/<person>_mask.png` (the upper-body clothing + arm
region any of the models may repaint). Pose uses the full frame.

Per-step curves (one value per denoising step):
  pose            OpenPose PCK of the frame's skeleton vs the final frame's,
                  averaged over 0.05/0.1/0.2 torso-length tolerances (pose.py).
  color           fraction of garment pixels whose CIE Lab color is within
                  COLOR_DELTA_E of the final frame (Delta-E 10 = clearly the
                  same color family; ~2 is the just-noticeable difference).
  structure       mean SSIM vs the final frame over garment pixels (grayscale).
  texture         amount of fine detail vs the final frame: ratio of the two
                  high-frequency FFT energy fractions, smaller/larger. 1 = as
                  much fine detail as the final garment, whatever it is.
  pattern         similarity of the garment's radial log-power spectrum to the
                  final frame's -- is the specific pattern scale (stripe period,
                  print repeat) already the one it ends with.
  print_fidelity  the same spectral similarity, but vs the source garment
                  photo (cropped to its cloth-mask): how much the print looks
                  like the actual garment. NOT referenced to the final frame,
                  so it can peak mid-trajectory and drop again.
  stability       SSIM vs the PREVIOUS step over garment pixels: 1 = x0-hat no
                  longer changing. Undefined (NaN) at step 0.

Per-run summaries (per curve):
  auc             mean of the curve over all steps = normalized area under it.
                  High = the frame is close to its final state for most of
                  the trajectory, i.e. the model commits early.
  headstart       fraction of the trajectory still REMAINING when the curve
                  reaches SETTLE_THRESHOLD and stays there until the end.
                  1 = settled from the very first step, 0 = only at the end.
  final           last value (only meaningful for print_fidelity: how well the
                  final try-on reproduces the garment's pattern).

These replace the earlier min-max-normalized "onsets", which (a) turned one
noisy first step (IDM-VTON's x0-hat at t=981 is nearly pure noise) into an
"everything settles at step 1" artifact and (b) measured when a curve made
its last small corrections rather than how close it already was.

Usage:
    python metrics.py --run-dir outputs/idm_pair1_solid
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose import pose_pck, run_keypoints  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS_VERSION = 2

ANALYSIS_SIZE = (384, 512)      # (w, h) every frame is resampled to
COLOR_DELTA_E = 10.0
SPECTRAL_SIZE = (128, 128)      # garment crops are resized to this before spectral profiles
PROFILE_BINS = 20
HIGH_FREQ_CUT = 0.4             # radial fraction above which FFT energy counts as "fine detail"
SETTLE_THRESHOLD = 0.8

# (key, plot title, y-label, has headstart) -- the single source of truth
# for which curves exist; compare_trajectories.py imports this.
AXES = (
    ("pose", "POSE", "OpenPose PCK vs final skeleton", True),
    ("color", "COLOR", f"garment pixels within dE {COLOR_DELTA_E:g} of final", True),
    ("structure", "STRUCTURE", "SSIM vs final (garment)", True),
    ("texture", "TEXTURE", "fine-detail amount vs final", True),
    ("pattern", "PATTERN", "spectral pattern vs final", True),
    ("print_fidelity", "PRINT FIDELITY", "spectral pattern vs garment photo", False),
    ("stability", "STABILITY", "SSIM vs previous step (garment)", True),
)


# --------------------------------------------------------------------- inputs

def resolve_input(path_str: str) -> Path:
    """run_config.json stores absolute input paths from the machine the run
    happened on; when outputs are copied elsewhere (cluster -> laptop), fall
    back to the same path under this checkout's data/ dir."""
    path = Path(path_str)
    if path.exists() or "data" not in path.parts:
        return path
    parts = path.parts
    last_data = len(parts) - 1 - parts[::-1].index("data")
    return REPO_ROOT.joinpath(*parts[last_data:])


def load_rgb(path: Path, size=ANALYSIS_SIZE) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img).astype(np.float32) / 255.0


def load_mask(path: Path, size) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize(size, Image.NEAREST)) > 127


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


# ------------------------------------------------------------------- spectra

def masked_crop(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Bounding-box crop of the masked region with the non-masked pixels
    inside the box filled with the masked mean, so the FFT sees the garment
    rather than a hard mask edge or background."""
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop, m = gray[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1]
    crop[~m] = crop[m].mean()
    return crop


def _radius(shape) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2) / np.sqrt(2)


def power_spectrum(gray: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean()))) ** 2


def high_freq_fraction(gray: np.ndarray) -> float:
    power = power_spectrum(gray)
    total = power.sum()
    return float(power[_radius(gray.shape) > HIGH_FREQ_CUT].sum() / total) if total > 0 else 0.0


def radial_profile(gray: np.ndarray) -> np.ndarray:
    """Fraction of FFT power in PROFILE_BINS equal-width radial bands: a
    fingerprint of the pattern's characteristic scale."""
    small = np.asarray(Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8))
                       .resize(SPECTRAL_SIZE, Image.LANCZOS)).astype(np.float32) / 255.0
    power, r = power_spectrum(small), _radius(SPECTRAL_SIZE[::-1])
    edges = np.linspace(0.0, 1.0, PROFILE_BINS + 1)
    prof = np.array([power[(r > edges[i]) & (r <= edges[i + 1])].sum() for i in range(PROFILE_BINS)])
    return prof / prof.sum() if prof.sum() > 0 else prof


def profile_similarity(p: np.ndarray, q: np.ndarray) -> float:
    """exp(-mean |log10 p - log10 q|): 1 for identical spectra. Compared in
    log space because power falls off by orders of magnitude with frequency;
    a linear comparison is dominated by the lowest bin and barely moves."""
    return float(np.exp(-np.abs(np.log10(p + 1e-8) - np.log10(q + 1e-8)).mean()))


# ------------------------------------------------------------- per-step scores

def color_agreement(lab: np.ndarray, ref_lab: np.ndarray, mask: np.ndarray) -> float:
    delta_e = np.linalg.norm(lab - ref_lab, axis=-1)
    return float((delta_e[mask] < COLOR_DELTA_E).mean())


def masked_ssim(gray: np.ndarray, ref_gray: np.ndarray, mask: np.ndarray) -> float:
    _, smap = structural_similarity(gray, ref_gray, data_range=1.0, full=True)
    return float(np.clip(smap[mask].mean(), 0.0, 1.0))


def detail_ratio(hf: float, ref_hf: float) -> float:
    lo, hi = sorted((hf, ref_hf))
    return 1.0 if hi <= 0 else float(lo / hi)


# ---------------------------------------------------------------- summaries

def summarize(curve, has_headstart: bool) -> dict:
    v = np.asarray(curve, dtype=np.float64)
    finite = np.isfinite(v)
    out = {"auc": float(np.nanmean(v)) if finite.any() else None,
           "final": float(v[-1]) if finite[-1] else None}
    if has_headstart and finite.any():
        n = len(v)
        ok = np.where(finite, v >= SETTLE_THRESHOLD, True)  # NaN (stability step 0) doesn't break a streak
        settled = None
        for i in range(n):
            if ok[i:].all() and finite[i:].any():
                settled = i
                break
        out["headstart"] = 0.0 if settled is None else 1.0 - settled / (n - 1)
    return out


def garment_mask_path(person_path: Path) -> Path:
    """The dataset's agnostic mask for this person -- identical for every model."""
    return person_path.parents[1] / "agnostic-mask" / f"{person_path.stem}_mask.png"


def compute_run_metrics(run_dir: Path) -> dict:
    config = json.loads((run_dir / "run_config.json").read_text())
    timesteps = config["timesteps"]
    frame_paths = sorted((run_dir / "frames_x0").glob("step_*.png"))
    assert len(frame_paths) == len(timesteps), f"{len(frame_paths)} frames vs {len(timesteps)} timesteps"

    person_path = resolve_input(config["inputs"]["person"])
    cloth_path = resolve_input(config["inputs"]["cloth"])
    mask = load_mask(garment_mask_path(person_path), ANALYSIS_SIZE)
    cloth_mask = load_mask(cloth_path.parents[1] / "cloth-mask" / cloth_path.name, Image.open(cloth_path).size)

    frames = [load_rgb(p) for p in frame_paths]
    grays = [to_gray(f) for f in frames]
    final_lab, final_gray = rgb2lab(frames[-1]), grays[-1]
    final_crop = masked_crop(final_gray, mask)
    final_hf, final_profile = high_freq_fraction(final_crop), radial_profile(final_crop)
    cloth_gray = to_gray(np.asarray(Image.open(cloth_path).convert("RGB")).astype(np.float32) / 255.0)
    cloth_profile = radial_profile(masked_crop(cloth_gray, cloth_mask))

    kps = run_keypoints(run_dir, frame_paths, person_path)
    curves = {k: [] for k, *_ in AXES}
    curves["pose"] = [pose_pck(kp, kps["frames"][-1]) for kp in kps["frames"]]
    for i, (rgb, gray) in enumerate(zip(frames, grays)):
        crop = masked_crop(gray, mask)
        profile = radial_profile(crop)
        curves["color"].append(color_agreement(rgb2lab(rgb), final_lab, mask))
        curves["structure"].append(masked_ssim(gray, final_gray, mask))
        curves["texture"].append(detail_ratio(high_freq_fraction(crop), final_hf))
        curves["pattern"].append(profile_similarity(profile, final_profile))
        curves["print_fidelity"].append(profile_similarity(profile, cloth_profile))
        curves["stability"].append(float("nan") if i == 0 else masked_ssim(gray, grays[i - 1], mask))

    # Supplementary, not an axis: does the try-on keep the person's own pose.
    pose_vs_person = [pose_pck(kp, kps["person"]) for kp in kps["frames"]]

    return {
        "version": METRICS_VERSION,
        "run_dir": str(run_dir),
        "model": config.get("model"),
        "timesteps": timesteps,
        "settle_threshold": SETTLE_THRESHOLD,
        "curves": curves,
        "pose_vs_person": pose_vs_person,
        "summary": {k: summarize(curves[k], hs) for k, _, _, hs in AXES},
    }


# ------------------------------------------------------------------ plotting

AXIS_COLORS = {"pose": "#3b82f6", "color": "#10b981", "structure": "#f59e0b", "texture": "#f97316",
               "pattern": "#a855f7", "print_fidelity": "#ec4899", "stability": "#64748b"}


def plot_run(result: dict, out_path: Path):
    n = len(result["timesteps"])
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (key, title, ylabel, has_hs) in zip(axes.flat, AXES):
        color, s = AXIS_COLORS[key], result["summary"][key]
        ax.plot(range(n), result["curves"][key], color=color)
        if key == "pose":
            ax.plot(range(n), result["pose_vs_person"], color=color, linestyle="--", alpha=0.6, label="vs input person")
            ax.legend(fontsize=7, loc="lower right")
        if has_hs:
            ax.axhline(SETTLE_THRESHOLD, color="gray", linestyle=":", linewidth=0.8)
            if s.get("headstart") is not None:
                ax.axvline((1 - s["headstart"]) * (n - 1), color=color, linestyle=":", alpha=0.8)
        stats = f"auc {s['auc']:.2f}" + (f" | headstart {s['headstart']:.2f}" if s.get("headstart") is not None else "") \
            + (f" | final {s['final']:.2f}" if key == "print_fidelity" else "")
        ax.set_title(f"{title}\n{stats}", color=color, fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("denoising step")
        ax.set_ylabel(ylabel, fontsize=8)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, 0.5, "all scores: higher = better / closer\n"
                       f"dotted line: settle threshold {SETTLE_THRESHOLD}\n"
                       "auc: mean over steps (high = commits early)\n"
                       "headstart: share of steps left when settled", fontsize=9, va="center")
    fig.suptitle(f"{result['model']} -- {Path(result['run_dir']).name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def load_or_compute(run_dir: Path, force: bool = False) -> dict:
    cached = run_dir / "metrics.json"
    if cached.exists() and not force:
        result = json.loads(cached.read_text())
        if result.get("version") == METRICS_VERSION:
            return result
    result = compute_run_metrics(run_dir)
    cached.write_text(json.dumps(result, indent=1))
    plot_run(result, run_dir / "metrics.png")
    for stale in ("metrics_masked.json", "metrics_masked.png"):  # pre-v2 outputs
        (run_dir / stale).unlink(missing_ok=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--force", action="store_true", help="recompute even if metrics.json is current")
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    result = load_or_compute(run_dir, force=args.force)
    for key, s in result["summary"].items():
        print(f"[{run_dir.name}] {key:15} " + "  ".join(f"{k} {v:.2f}" for k, v in s.items() if v is not None))
    print(f"Wrote {run_dir / 'metrics.json'} and .png")


if __name__ == "__main__":
    main()
