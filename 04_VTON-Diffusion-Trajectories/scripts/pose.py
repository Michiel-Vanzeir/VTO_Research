"""
OpenPose body keypoints for the POSE axis of metrics.py.

Uses the ControlNet-annotator port of CMU OpenPose's 18-keypoint (COCO
ordering) body model that OOTDiffusion and IDM-VTON both vendor
(repos/OOTDiffusion/preprocess/openpose/annotator/openpose/body.py -- the
two copies are identical apart from a comment and the checkpoint dir). It
is the same detector family that produced the VITON-HD dataset's
`openpose_json` annotations and that OOTDiffusion runs live at inference,
so "pose" here means exactly the skeleton these try-on models are
conditioned on / trained against.

Keypoints are stored as (x/H, y/H, confidence) -- both coordinates divided
by image HEIGHT, so distances stay isotropic and are comparable between
CatVTON's 384x512 frames and OOTDiffusion/IDM-VTON's 768x1024 frames (all
3:4). Missing keypoints are NaN with confidence 0.

COCO-18 order: 0 nose, 1 neck, 2 r-shoulder, 3 r-elbow, 4 r-wrist,
5 l-shoulder, 6 l-elbow, 7 l-wrist, 8 r-hip, 9 r-knee, 10 r-ankle,
11 l-hip, 12 l-knee, 13 l-ankle, 14 r-eye, 15 l-eye, 16 r-ear, 17 l-ear.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OPENPOSE_DIR = REPO_ROOT / "repos" / "OOTDiffusion" / "preprocess" / "openpose"
BODY_WEIGHTS = REPO_ROOT / "repos" / "OOTDiffusion" / "checkpoints" / "openpose" / "ckpts" / "body_pose_model.pth"

N_KEYPOINTS = 18
NECK, R_SHOULDER, L_SHOULDER, R_HIP, L_HIP = 1, 2, 5, 8, 11
# PCK is averaged over several tolerances (fractions of torso length) so
# the curve moves smoothly instead of in 1/18 jumps at a single threshold.
PCK_THRESHOLDS = (0.05, 0.1, 0.2)

_body = None


def get_body():
    global _body
    if _body is None:
        if not BODY_WEIGHTS.exists():
            raise FileNotFoundError(
                f"{BODY_WEIGHTS} missing -- run cluster/setup.sh (or download "
                "body_pose_model.pth from the lllyasviel/Annotators HF repo into that dir)"
            )
        sys.path.insert(0, str(OPENPOSE_DIR))
        from annotator.openpose.body import Body  # noqa: E402
        _body = Body(str(BODY_WEIGHTS))
    return _body


def detect_keypoints(rgb: np.ndarray) -> np.ndarray:
    """(18, 3) array of (x/H, y/H, confidence) for the most complete person
    detected in an RGB uint8 image; NaN rows for undetected keypoints."""
    h = rgb.shape[0]
    candidate, subset = get_body()(rgb[:, :, ::-1].copy())  # Body expects BGR
    kp = np.full((N_KEYPOINTS, 3), np.nan, dtype=np.float64)
    kp[:, 2] = 0.0
    if len(subset) == 0:
        return kp
    # subset rows: 18 candidate indices, total score, keypoint count
    person = subset[np.lexsort((subset[:, -2], subset[:, -1]))[-1]]
    for k in range(N_KEYPOINTS):
        idx = int(person[k])
        if idx >= 0:
            x, y, score = candidate[idx][:3]
            kp[k] = (x / h, y / h, score)
    return kp


def detect_keypoints_file(path: Path) -> np.ndarray:
    return detect_keypoints(np.asarray(Image.open(path).convert("RGB")))


def torso_length(kp: np.ndarray) -> float:
    """Neck-to-mid-hip distance; falls back to 1.5x shoulder width (typical
    torso/shoulder ratio) when no hip is visible. NaN if neither works."""
    hips = kp[[R_HIP, L_HIP], :2]
    hips = hips[~np.isnan(hips[:, 0])]
    if not np.isnan(kp[NECK, 0]) and len(hips):
        return float(np.linalg.norm(kp[NECK, :2] - hips.mean(axis=0)))
    if not np.isnan(kp[R_SHOULDER, 0]) and not np.isnan(kp[L_SHOULDER, 0]):
        return 1.5 * float(np.linalg.norm(kp[R_SHOULDER, :2] - kp[L_SHOULDER, :2]))
    return float("nan")


def _normalized_errors(kp: np.ndarray, ref: np.ndarray):
    """Per-reference-keypoint distance in torso units (inf where `kp` misses
    a keypoint the reference has), or None if the reference is unusable."""
    ref_valid = ~np.isnan(ref[:, 0])
    scale = torso_length(ref)
    if not ref_valid.any() or not np.isfinite(scale) or scale <= 0:
        return None
    err = np.full(N_KEYPOINTS, np.inf)
    both = ref_valid & ~np.isnan(kp[:, 0])
    err[both] = np.linalg.norm(kp[both, :2] - ref[both, :2], axis=1) / scale
    return err[ref_valid]


def pose_pck(kp: np.ndarray, ref: np.ndarray) -> float:
    """Percentage of Correct Keypoints vs `ref`, averaged over PCK_THRESHOLDS
    (torso-length fractions). A keypoint present in `ref` but missing in
    `kp` counts as incorrect, so a frame where OpenPose cannot yet find a
    body at all scores 0. In [0, 1]; NaN if `ref` has no usable skeleton."""
    err = _normalized_errors(kp, ref)
    if err is None:
        return float("nan")
    return float(np.mean([(err <= a).mean() for a in PCK_THRESHOLDS]))


def pose_kp_error(kp: np.ndarray, ref: np.ndarray) -> float:
    """Mean distance (torso units) over keypoints found in both; NaN if none."""
    err = _normalized_errors(kp, ref)
    if err is None or not np.isfinite(err).any():
        return float("nan")
    return float(err[np.isfinite(err)].mean())


def run_keypoints(run_dir: Path, frame_paths, person_path: Path) -> dict:
    """Keypoints for every x0-hat frame plus the input person image, cached
    in run_dir/pose_keypoints.json -- detection is the slow part of the
    metrics pass and doesn't depend on --crop-to-mask, so both passes share it."""
    cache = run_dir / "pose_keypoints.json"
    names = [p.name for p in frame_paths]
    if cache.exists():
        data = json.loads(cache.read_text())
        if data.get("frames") == names:
            return {"frames": np.array(data["keypoints"], dtype=np.float64),
                    "person": np.array(data["person"], dtype=np.float64)}
    frames = np.stack([detect_keypoints_file(p) for p in frame_paths])
    person = detect_keypoints_file(person_path)
    cache.write_text(json.dumps({
        "format": "COCO-18 (x/H, y/H, confidence), NaN = not detected",
        "frames": names,
        "keypoints": frames.tolist(),
        "person": person.tolist(),
    }))
    return {"frames": frames, "person": person}
