"""
Quantitative per-step analysis of a captured diffusion trajectory, along
four independent axes of "what has this model committed to yet":

  - POSE:    the body skeleton. OpenPose (18 COCO body keypoints, see
             pose.py) is run on every x0-hat frame and scored as PCK
             against the skeleton detected in the run's own final frame --
             so it only moves once the frame contains a body OpenPose can
             actually parse, with limbs where they end up, and is blind to
             color/texture. (The earlier downsampled-SSIM proxy for this is
             kept as `layout_ssim_to_final`, a coarse silhouette signal.)
  - COLOR:   the garment's color palette. Measured as (negative) mean Lab
             color distance to the final frame -- independent of SSIM's
             luminance/structure bias, so a frame can have the right color
             long before or after it has the right shape.
  - TEXTURE: generic local high-frequency detail -- fabric weave, logo
             edges, any fine spatial structure -- regardless of whether it
             matches the source garment's specific pattern. Measured
             per-frame (no reference needed) from FFT high-frequency power
             fraction and a corroborating Sobel edge-density signal.
  - PRINT:   whether the *specific* pattern being reproduced (stripe
             period, print repeat, logo scale) actually matches the source
             cloth reference image, as opposed to merely "having texture".
             Measured by comparing the radial power-spectrum profile of
             the garment-region crop against the same profile computed on
             the source cloth image -- two images can have identical
             high_freq_frac (same *amount* of detail) while this differs
             (wrong detail).

Works on the frames_x0/ produced by catvton_trajectory.py,
ootd_trajectory.py or idmvton_trajectory.py -- all write the same
step_NNN.png + run_config.json layout (plus the mask and source cloth image
referenced from run_config["inputs"]), so this script is model-agnostic.

Metrics computed per step, on the x0-hat frame (optionally cropped to the
inpainting mask's bounding box for the color/texture axes, see
--crop-to-mask; the print axis always crops to the mask bbox, since it is
inherently about the garment region, and the pose axis always uses the full
frame, since OpenPose needs the whole body):
  - ssim_to_final, psnr_to_final: whole-frame similarity to the run's own
    last frame (kept for reference/back-compat; superseded as the
    structure signal by the coarser layout_ssim_to_final below, which is
    less contaminated by fine local detail).
  - pose_pck_to_final: OpenPose PCK of the frame's skeleton vs the final
    frame's skeleton, averaged over tolerances of 0.05/0.1/0.2 torso
    lengths; keypoints missing from the frame count as wrong -- the POSE
    signal.
  - pose_pck_to_person: the same score vs the skeleton detected on the
    input person photo (does the try-on keep the person's pose at all).
  - pose_kp_err_to_final: mean keypoint distance to the final skeleton, in
    torso lengths, over keypoints detected in both.
  - pose_n_keypoints: number of the 18 keypoints OpenPose finds.
  - layout_ssim_to_final: SSIM between a downsampled (32x24) version of the
    frame and of the final frame -- coarse silhouette/layout (this was the
    POSE signal before the switch to OpenPose).
  - color_dist_to_final: mean per-pixel Euclidean distance in CIE Lab to
    the final frame (stored as a *distance*, so it falls toward 0; the
    onset is computed from its negation) -- the COLOR signal.
  - high_freq_frac: fraction of 2D FFT power (excluding DC) falling in the
    outer 60% of the radial frequency spectrum. This is an absolute
    per-frame TEXTURE signal -- it does not need a reference frame, and
    rises only once fine spatial detail (weave, print, logo edges) is
    actually present in the decoded x0-hat, not just a blurry color blob.
  - edge_density: mean Sobel gradient magnitude, a spatial-domain
    corroborating signal for the same texture question (cheaper, more
    intuitive, less sensitive to FFT windowing artifacts than high_freq_frac).
  - print_pattern_dist: L2 distance between the radial power-spectrum
    profile (20 bins) of the mask-bbox garment crop and of the source
    cloth image resized to the same shape (stored as a *distance*, so the
    onset is computed from its negation) -- the PRINT signal.
  - step_delta: RMS pixel change vs. the previous step's x0-hat, i.e. how
    much the model's clean-image estimate is still moving.

"Onset" timesteps are then extracted from the normalized (min-max) curves
as the first 50%-of-total-rise crossing, interpolated between the two
bracketing steps for sub-step precision. Using a *relative* rise (not an
absolute threshold like ssim>=0.5) makes onsets comparable across runs
whose absolute metric ranges differ with image content. Unlike the other
three axes, print_pattern_dist is not a self-referential (vs. own final
frame) signal, so its curve need not be monotonically best at the last
step -- onset_from_curve's rise-then-sustain check handles that: it flags
where the curve settles, wherever that is, and returns None if it never
does.

Usage:
    python metrics.py --run-dir outputs/pair1_solid
    python metrics.py --run-dir outputs/pair1_solid --crop-to-mask
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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose import pose_kp_error, pose_pck, run_keypoints  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

LOW_HIGH_SPLIT = (0.15, 0.4)  # radial-frequency fractions separating low/mid/high bands
LAYOUT_DOWNSAMPLE_SIZE = (32, 24)  # (w, h) -- gross layout only, no fine local detail
PRINT_PROFILE_BINS = 20
PRINT_COMPARE_SIZE = (128, 128)  # common size so radial bins line up between crop and cloth ref
PCK_SATURATION_RANGE = 0.1  # PCK varying less than this over the whole run = pose fixed from the start


def load_frame(path: Path, bbox=None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if bbox is not None:
        img = img.crop(bbox)
    return np.asarray(img).astype(np.float32) / 255.0


def mask_bbox(mask_path: Path, pad_frac: float = 0.05) -> tuple:
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return (0, 0, w, h)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = mask.shape[1], mask.shape[0]
    padx, pady = int((x1 - x0) * pad_frac), int((y1 - y0) * pad_frac)
    return (max(0, x0 - padx), max(0, y0 - pady), min(w, x1 + padx), min(h, y1 + pady))


def spectral_bands(gray: np.ndarray) -> tuple:
    """Radially-binned fraction of FFT power (DC excluded) in low/mid/high bands."""
    f = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(f) ** 2
    h, w = gray.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()
    dc = (r_norm < 1e-6)
    total = power.sum() - power[dc].sum()
    if total <= 0:
        return 0.0, 0.0, 0.0
    lo_cut, hi_cut = LOW_HIGH_SPLIT
    low = power[(r_norm > 0) & (r_norm <= lo_cut)].sum() / total
    mid = power[(r_norm > lo_cut) & (r_norm <= hi_cut)].sum() / total
    high = power[r_norm > hi_cut].sum() / total
    return float(low), float(mid), float(high)


def edge_density(gray: np.ndarray) -> float:
    gy, gx = np.gradient(gray)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def downsample_gray(gray: np.ndarray, size=LAYOUT_DOWNSAMPLE_SIZE) -> np.ndarray:
    """Coarse (w,h)-sized version of a [0,1] grayscale frame, for the layout
    signal: block-level layout only, no fine local detail."""
    img = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8))
    img = img.resize(size, Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def layout_ssim(gray: np.ndarray, final_gray: np.ndarray) -> float:
    a, b = downsample_gray(gray), downsample_gray(final_gray)
    win_size = min(7, min(a.shape) - (1 - min(a.shape) % 2))  # largest odd <= min dim, capped at 7
    return float(structural_similarity(a, b, data_range=1.0, win_size=win_size))


def color_dist_lab(rgb: np.ndarray, final_rgb: np.ndarray) -> float:
    """Mean per-pixel Euclidean distance in CIE Lab -- the COLOR signal.
    Lab separates lightness from hue/chroma, so this tracks palette
    convergence independent of the luminance/structure bias in SSIM."""
    lab, final_lab = rgb2lab(rgb), rgb2lab(final_rgb)
    return float(np.sqrt(((lab - final_lab) ** 2).sum(axis=-1)).mean())


def radial_power_profile(gray: np.ndarray, n_bins: int = PRINT_PROFILE_BINS) -> np.ndarray:
    """Fraction of FFT power (DC excluded) in `n_bins` equal-width radial
    bands -- a spectral fingerprint of the pattern's characteristic scale
    (stripe period, print repeat, weave frequency), finer-grained than the
    3-band split in `spectral_bands` above."""
    f = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(f) ** 2
    h, w = gray.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()
    dc = r_norm < 1e-6
    total = power.sum() - power[dc].sum()
    profile = np.zeros(n_bins, dtype=np.float64)
    if total <= 0:
        return profile
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        sel = (~dc) & (r_norm > edges[i]) & (r_norm <= edges[i + 1])
        profile[i] = power[sel].sum() / total
    return profile


def load_gray_resized(path: Path, size=PRINT_COMPARE_SIZE) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size, Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def print_pattern_dist(garment_gray: np.ndarray, cloth_gray: np.ndarray) -> float:
    """L2 distance between the two images' radial power-spectrum profiles
    -- the PRINT signal: does the *specific* pattern (not just "how much
    detail") match the source cloth."""
    p1 = radial_power_profile(garment_gray)
    p2 = radial_power_profile(cloth_gray)
    return float(np.sqrt(np.mean((p1 - p2) ** 2)))


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


def resolve_mask_path(run_dir: Path, config: dict) -> Path:
    mask_path = run_dir / "mask.png"
    if not mask_path.exists():
        mask_path = resolve_input(config["inputs"]["mask"])
    return mask_path


def compute_run_metrics(run_dir: Path, crop_to_mask: bool = False) -> dict:
    config = json.loads((run_dir / "run_config.json").read_text())
    timesteps = config["timesteps"]
    frame_paths = sorted((run_dir / "frames_x0").glob("step_*.png"))
    assert len(frame_paths) == len(timesteps), f"{len(frame_paths)} frames vs {len(timesteps)} timesteps"

    # Garment-region bbox is always needed for the PRINT axis, regardless of
    # --crop-to-mask (which only controls whether the *other* axes look at
    # the whole frame or just the garment region).
    garment_bbox = mask_bbox(resolve_mask_path(run_dir, config))
    bbox = garment_bbox if crop_to_mask else None

    frames = [load_frame(p, bbox) for p in frame_paths]
    final = frames[-1]
    final_gray = to_gray(final)

    cloth_gray = load_gray_resized(resolve_input(config["inputs"]["cloth"]))
    garment_crops = frames if bbox == garment_bbox else [load_frame(p, garment_bbox) for p in frame_paths]

    ssim_to_final, psnr_to_final, high_freq_frac, mid_freq_frac, low_freq_frac = [], [], [], [], []
    edge_dens, step_delta, layout_ssim_to_final, color_dist_to_final, print_pattern_d = [], [0.0], [], [], []
    for i, frame in enumerate(frames):
        gray = to_gray(frame)
        ssim_to_final.append(float(structural_similarity(gray, final_gray, data_range=1.0)))
        psnr = peak_signal_noise_ratio(final, frame, data_range=1.0)
        psnr_to_final.append(100.0 if np.isinf(psnr) else float(psnr))  # cap identical-frame case (last step vs itself)
        low, mid, high = spectral_bands(gray)
        low_freq_frac.append(low)
        mid_freq_frac.append(mid)
        high_freq_frac.append(high)
        edge_dens.append(edge_density(gray))
        layout_ssim_to_final.append(layout_ssim(gray, final_gray))
        color_dist_to_final.append(color_dist_lab(frame, final))
        garment_gray = to_gray(garment_crops[i])
        print_pattern_d.append(print_pattern_dist(
            np.asarray(Image.fromarray((np.clip(garment_gray, 0, 1) * 255).astype(np.uint8))
                       .resize(PRINT_COMPARE_SIZE, Image.BILINEAR)).astype(np.float32) / 255.0,
            cloth_gray,
        ))
        if i > 0:
            step_delta.append(float(np.sqrt(np.mean((frame - frames[i - 1]) ** 2))))

    # POSE: OpenPose on the full (uncropped) frames, whatever --crop-to-mask says.
    kps = run_keypoints(run_dir, frame_paths, resolve_input(config["inputs"]["person"]))
    final_kp = kps["frames"][-1]
    pose_pck_to_final = [pose_pck(kp, final_kp) for kp in kps["frames"]]
    pose_pck_to_person = [pose_pck(kp, kps["person"]) for kp in kps["frames"]]
    pose_kp_err_to_final = [pose_kp_error(kp, final_kp) for kp in kps["frames"]]
    pose_n_keypoints = [int((~np.isnan(kp[:, 0])).sum()) for kp in kps["frames"]]

    color_similarity = [-d for d in color_dist_to_final]
    print_similarity = [-d for d in print_pattern_d]

    # step_delta falls as the estimate stabilizes, rather than rising like
    # the other curves -- negate before onset_from_curve, same trick as
    # color/print. This answers "has x0-hat stopped still revising itself"
    # (distinct from "has it converged to the specific final value").
    onsets = {
        # PCK moves in whole-keypoint steps and is often ~1.0 from step 0 for
        # inpainting models (the unmasked body pins the skeleton), so a <0.1
        # total range is reported as "settled at step 0" rather than min-max
        # stretching detector jitter into a fake onset. pose_refine_onset is
        # the sub-threshold version: when joint positions stop drifting.
        "pose_onset": onset_from_curve(pose_pck_to_final, timesteps, saturate_below=PCK_SATURATION_RANGE),
        "pose_refine_onset": onset_from_curve([-e for e in pose_kp_err_to_final], timesteps),
        "color_onset": onset_from_curve(color_similarity, timesteps),
        "texture_onset": onset_from_curve(high_freq_frac, timesteps),
        "print_onset": onset_from_curve(print_similarity, timesteps),
        "stability_onset": onset_from_curve([-d for d in step_delta], timesteps),
        "structure_onset": onset_from_curve(ssim_to_final, timesteps),
        "edge_onset": onset_from_curve(edge_dens, timesteps),
        "layout_onset": onset_from_curve(layout_ssim_to_final, timesteps),
    }

    return {
        "run_dir": str(run_dir),
        "model": config.get("model"),
        "timesteps": timesteps,
        "crop_to_mask": crop_to_mask,
        "metrics": {
            "ssim_to_final": ssim_to_final,
            "psnr_to_final": psnr_to_final,
            "pose_pck_to_final": pose_pck_to_final,
            "pose_pck_to_person": pose_pck_to_person,
            "pose_kp_err_to_final": pose_kp_err_to_final,
            "pose_n_keypoints": pose_n_keypoints,
            "layout_ssim_to_final": layout_ssim_to_final,
            "color_dist_to_final": color_dist_to_final,
            "low_freq_frac": low_freq_frac,
            "mid_freq_frac": mid_freq_frac,
            "high_freq_frac": high_freq_frac,
            "edge_density": edge_dens,
            "print_pattern_dist": print_pattern_d,
            "step_delta": step_delta,
        },
        "onsets": onsets,
    }


def onset_from_curve(values, timesteps, rise_frac: float = 0.5, sustain: int = 3,
                     saturate_below: float = None) -> dict:
    """First step (interpolated) where the min-max-normalized curve crosses
    `rise_frac` of its own total rise AND stays there for `sustain` steps.
    The sustain requirement matters because x0-hat at very high noise
    levels can be a poor, high-variance extrapolation (dividing by a small
    alpha_bar_t amplifies noise_pred error) -- occasionally spiking a
    texture-like signal for a step or two before the trajectory actually
    settles. Without it, a single early spike gets reported as "texture
    from step 0", which is a measurement artifact, not structure/texture
    formation. Returns None if the curve never rises enough to cross, or
    never sustains the crossing (e.g. it's flat, noisy throughout, or falls
    instead).

    `saturate_below`: for bounded curves (pose PCK), if the total range is
    smaller than this the curve is treated as already settled at step 0
    (returned with "saturated": True) instead of normalizing noise."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).any():
        return None  # e.g. pose PCK when OpenPose found no body in the final frame
    values = np.where(np.isfinite(values), values, np.nanmin(values))
    vmin, vmax = values.min(), values.max()
    if saturate_below is not None and vmax - vmin < saturate_below:
        return {"step": 0.0, "timestep": float(timesteps[0]), "saturated": True}
    if vmax - vmin < 1e-8:
        return None
    norm = (values - vmin) / (vmax - vmin)
    n = len(norm)
    candidates = np.where(norm >= rise_frac)[0]
    onset_i = None
    for c in candidates:
        window = norm[c:min(c + sustain, n)]
        if window.min() >= rise_frac:
            onset_i = int(c)
            break
    if onset_i is None:
        return None
    i = onset_i
    if i == 0:
        step_frac, t_interp = 0.0, float(timesteps[0])
    else:
        v0, v1 = norm[i - 1], norm[i]
        frac = (rise_frac - v0) / (v1 - v0) if v1 != v0 else 0.0
        step_frac = (i - 1) + frac
        t0, t1 = timesteps[i - 1], timesteps[i]
        t_interp = t0 + frac * (t1 - t0)
    return {"step": step_frac, "timestep": t_interp}


AXIS_SPECS = (
    # (title, onset_key, color, [(metric_key, label, linestyle), ...])
    ("POSE", "pose_onset", "#3b82f6", [
        ("pose_pck_to_final", "OpenPose PCK vs final skeleton", "-"),
        ("pose_pck_to_person", "OpenPose PCK vs input person", "--"),
    ]),
    ("COLOR", "color_onset", "#10b981", [("color_dist_to_final", "Lab distance to final", "-")]),
    ("TEXTURE", "texture_onset", "#f97316", [
        ("high_freq_frac", "high-freq FFT fraction", "-"),
        ("edge_density", "edge density (corroborating)", "--"),
    ]),
    ("PRINT", "print_onset", "#a855f7", [("print_pattern_dist", "spectral distance to cloth ref", "-")]),
    ("STABILITY", "stability_onset", "#64748b", [("step_delta", "RMS change vs previous step", "-")]),
    ("POSE REFINE", "pose_refine_onset", "#0ea5e9", [("pose_kp_err_to_final", "keypoint error vs final (torso lengths)", "-")]),
)


def plot_run(result: dict, out_path: Path):
    m = result["metrics"]
    steps = np.arange(len(m["ssim_to_final"]))
    timesteps = result["timesteps"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (title, onset_key, color, series) in zip(axes.flat, AXIS_SPECS):
        for metric_key, label, ls in series:
            ax.plot(steps, m[metric_key], color=color, linestyle=ls, alpha=1.0 if ls == "-" else 0.6, label=label)
        onset = result["onsets"][onset_key]
        if onset is not None:
            ax.axvline(onset["step"], color=color, linestyle=":", alpha=0.8)
            ax.annotate(f"t={onset['timestep']:.0f}", (onset["step"], ax.get_ylim()[0]),
                        color=color, fontsize=8, rotation=90, va="bottom")
        if title == "POSE":
            ax.set_ylim(-0.02, 1.05)  # PCK is a fraction; autoscaling blows single-keypoint flicker up to full height
        ax.set_title(title, color=color, fontsize=11)
        ax.set_xlabel(f"step (t: {timesteps[0]}->{timesteps[-1]})")
        ax.legend(fontsize=7, loc="best")
    for ax in axes.flat[len(AXIS_SPECS):]:
        ax.axis("off")

    fig.suptitle(f"{result['model']} -- {Path(result['run_dir']).name}"
                 + (" (mask crop)" if result["crop_to_mask"] else ""))
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--crop-to-mask", action="store_true")
    args = p.parse_args()
    run_dir = Path(args.run_dir)

    result = compute_run_metrics(run_dir, crop_to_mask=args.crop_to_mask)
    suffix = "_masked" if args.crop_to_mask else ""
    (run_dir / f"metrics{suffix}.json").write_text(json.dumps(result, indent=2))
    plot_run(result, run_dir / f"metrics{suffix}.png")

    for key, onset in result["onsets"].items():
        msg = f"step {onset['step']:.1f} (t={onset['timestep']:.0f})" if onset else "no clear crossing"
        print(f"[{run_dir.name}] {key}: {msg}")
    print(f"Wrote {run_dir / f'metrics{suffix}.json'} and .png")


if __name__ == "__main__":
    main()
