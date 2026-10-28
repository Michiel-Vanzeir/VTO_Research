"""
Per-step analysis of one captured diffusion trajectory (frames_x0/ written by
catvton_trajectory.py, ootd_trajectory.py or idmvton_trajectory.py).

EVERY SCORE IS IN [0, 1] AND HIGHER IS ALWAYS BETTER: for per-step curves
"higher" means "closer to where this run ends up"; for the per-run
summaries it means "committed earlier". Nothing needs to be mentally
inverted when reading plots or tables.

To keep the three models comparable, every frame is first resampled to one
analysis resolution (ANALYSIS_SIZE; CatVTON's native 384x512), and all
garment-region scores use the SAME pixel mask for every model: the VITON-HD
dataset's `agnostic-mask/<person>_mask.png` (the upper-body clothing + arm
region any of the models may repaint). Pose uses the full frame.

Per-step curves (one value per denoising step):
  pose            OpenPose PCK of the frame's skeleton vs the final frame's,
                  averaged over 0.05/0.1/0.2 torso-length tolerances (pose.py).
  color           mean over garment pixels of a per-pixel score that falls
                  continuously with the CIE Lab Delta-E to the final frame:
                  1 below COLOR_DELTA_E[0] (2.5 ~ just noticeable), 0 above
                  COLOR_DELTA_E[1] (20 = other color), linear in log Delta-E
                  in between. Equals the share of pixels within Delta-E t,
                  averaged over all t in that range spaced evenly in log, so
                  it never jumps when a pixel crosses a cutoff.
                  Both images are smoothed (sigma COLOR_SIGMA px) first, so it
                  compares the color of regions, not of individual pixels
                  (which would also react to noise and shifted edges).
  structure       mean SSIM vs the final frame over garment pixels, on
                  grayscale images smoothed with sigma STRUCTURE_SIGMA px so
                  noise and fine detail don't count as structure.
  texture         amount of fine detail vs the final frame: ratio of the two
                  energy fractions in the mid-high spatial-frequency band
                  TEXTURE_BAND, smaller/larger. 1 = as much fine detail as the
                  final garment. The band deliberately excludes the highest
                  frequencies, where pixel noise and resampling live.
  pattern         similarity of the garment's radial log-power spectrum to the
                  final frame's -- is the specific pattern scale (stripe period,
                  print repeat) already the one it ends with.
  stability       SSIM vs the PREVIOUS step over garment pixels: 1 = x0-hat no
                  longer changing. Undefined (NaN) at step 0.

Per-run summaries (per curve):
  auc             mean of the curve over all steps = normalized area under it.
                  High = the frame is close to its final state for most of
                  the trajectory, i.e. the model commits early.

The axis definitions were chosen with validate_axes.py (controlled blur /
color-shift / warp / noise perturbations of final images): color, texture
and structure each react mainly to their own perturbation; pattern reacts
mainly to blur, i.e. it is NOT independent of texture.

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
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose import pose_pck, run_keypoints  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS_VERSION = 4

ANALYSIS_SIZE = (384, 512)      # (w, h) every frame is resampled to
COLOR_DELTA_E = (2.5, 20.0)     # (full score below, zero score above)
COLOR_SIGMA = 4.0
STRUCTURE_SIGMA = 2.0
TEXTURE_BAND = (0.1, 0.3)         # radial frequency band (1 = spectrum corner) that counts as "fine detail"
SPECTRAL_SIZE = (128, 128)      # garment crops are resized to this before spectral profiles
PROFILE_BINS = 20

# (key, plot title, y-label) -- the single source of truth for which curves
# exist; compare_trajectories.py imports this.
AXES = (
    ("pose", "Pose", "OpenPose PCK vs final skeleton"),
    ("color", "Colour", "colour agreement vs final (dE 2.5-20)"),
    ("structure", "Structure", "coarse SSIM vs final (garment)"),
    ("texture", "Texture", "fine-detail amount vs final"),
    ("pattern", "Pattern", "spectral pattern vs final"),
    ("stability", "Stability", "SSIM vs previous step (garment)"),
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


def smooth_rgb(rgb: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack([gaussian_filter(rgb[..., c], sigma) for c in range(3)], -1)


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


def detail_fraction(gray: np.ndarray) -> float:
    """Share of FFT energy in the TEXTURE_BAND of radial frequencies."""
    power = power_spectrum(gray)
    total = power.sum()
    r = _radius(gray.shape)
    lo, hi = TEXTURE_BAND
    return float(power[(r > lo) & (r <= hi)].sum() / total) if total > 0 else 0.0


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
    delta_e = np.linalg.norm(lab - ref_lab, axis=-1)[mask]
    lo, hi = COLOR_DELTA_E
    score = np.log(hi / np.maximum(delta_e, 1e-12)) / np.log(hi / lo)
    return float(np.clip(score, 0.0, 1.0).mean())


def masked_ssim(gray: np.ndarray, ref_gray: np.ndarray, mask: np.ndarray, sigma: float = 0.0) -> float:
    if sigma:
        gray, ref_gray = gaussian_filter(gray, sigma), gaussian_filter(ref_gray, sigma)
    _, smap = structural_similarity(gray, ref_gray, data_range=1.0, full=True)
    return float(np.clip(smap[mask].mean(), 0.0, 1.0))


def detail_ratio(hf: float, ref_hf: float) -> float:
    lo, hi = sorted((hf, ref_hf))
    return 1.0 if hi <= 0 else float(lo / hi)


# ---------------------------------------------------------------- summaries

def summarize(curve) -> dict:
    v = np.asarray(curve, dtype=np.float64)
    return {"auc": float(np.nanmean(v)) if np.isfinite(v).any() else None}


def garment_mask_path(person_path: Path) -> Path:
    """The dataset's agnostic mask for this person -- identical for every model."""
    return person_path.parents[1] / "agnostic-mask" / f"{person_path.stem}_mask.png"


class Reference:
    """Precomputed per-image data of the frame everything is compared against
    (the run's final frame), so each step only computes its own side."""

    def __init__(self, rgb: np.ndarray, mask: np.ndarray):
        self.mask = mask
        self.lab = rgb2lab(smooth_rgb(rgb, COLOR_SIGMA))
        self.gray = to_gray(rgb)
        crop = masked_crop(self.gray, mask)
        self.detail = detail_fraction(crop)
        self.profile = radial_profile(crop)


def scores_vs_reference(rgb: np.ndarray, ref: Reference) -> dict:
    """The four final-referenced garment axes for one frame. Shared by
    compute_run_metrics and validate_axes.py, so the validation tests exactly
    the code that produces the results."""
    gray = to_gray(rgb)
    crop = masked_crop(gray, ref.mask)
    return {
        "color": color_agreement(rgb2lab(smooth_rgb(rgb, COLOR_SIGMA)), ref.lab, ref.mask),
        "structure": masked_ssim(gray, ref.gray, ref.mask, STRUCTURE_SIGMA),
        "texture": detail_ratio(detail_fraction(crop), ref.detail),
        "pattern": profile_similarity(radial_profile(crop), ref.profile),
    }


def compute_run_metrics(run_dir: Path) -> dict:
    config = json.loads((run_dir / "run_config.json").read_text())
    timesteps = config["timesteps"]
    frame_paths = sorted((run_dir / "frames_x0").glob("step_*.png"))
    assert len(frame_paths) == len(timesteps), f"{len(frame_paths)} frames vs {len(timesteps)} timesteps"

    person_path = resolve_input(config["inputs"]["person"])
    mask = load_mask(garment_mask_path(person_path), ANALYSIS_SIZE)

    frames = [load_rgb(p) for p in frame_paths]
    grays = [to_gray(f) for f in frames]
    ref = Reference(frames[-1], mask)

    kps = run_keypoints(run_dir, frame_paths, person_path)
    curves = {k: [] for k, *_ in AXES}
    curves["pose"] = [pose_pck(kp, kps["frames"][-1]) for kp in kps["frames"]]
    for i, (rgb, gray) in enumerate(zip(frames, grays)):
        for key, value in scores_vs_reference(rgb, ref).items():
            curves[key].append(value)
        curves["stability"].append(float("nan") if i == 0 else masked_ssim(gray, grays[i - 1], mask))

    # Supplementary, not an axis: does the try-on keep the person's own pose.
    pose_vs_person = [pose_pck(kp, kps["person"]) for kp in kps["frames"]]

    return {
        "version": METRICS_VERSION,
        "run_dir": str(run_dir),
        "model": config.get("model"),
        "timesteps": timesteps,
        "curves": curves,
        "pose_vs_person": pose_vs_person,
        "summary": {k: summarize(curves[k]) for k, *_ in AXES},
    }


# ------------------------------------------------------------------ plotting

AXIS_COLORS = {"pose": "#3b82f6", "color": "#10b981", "structure": "#f59e0b", "texture": "#f97316",
               "pattern": "#a855f7", "stability": "#64748b"}


def plot_run(result: dict, out_path: Path):
    n = len(result["timesteps"])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (key, title, ylabel) in zip(axes.flat, AXES):
        color = AXIS_COLORS[key]
        ax.plot(range(n), result["curves"][key], color=color)
        if key == "pose":
            ax.plot(range(n), result["pose_vs_person"], color=color, linestyle="--", alpha=0.6, label="vs input person")
            ax.legend(fontsize=7, loc="lower right")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("denoising step")
        ax.set_ylabel(ylabel, fontsize=8)
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
