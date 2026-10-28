"""
Standard final-image scores (FID, KID) of the 48 curated runs per model.

The runs use the unpaired VITON-HD protocol, so there is no ground-truth image
of the person wearing the garment and SSIM / LPIPS are not defined. FID and KID
compare the distribution of the 48 final images of a model with the 2,032 real
person images of the VITON-HD test split. Every image (real and generated) is
first resized to the common 384x512 evaluation frame of the paper, so the
models' different native resolutions do not change the score.

With only 48 generated images FID is strongly biased upwards, so the numbers
are only comparable within this table; KID is unbiased and is the one to read.

Usage:
    python scripts/final_image_metrics.py            # needs torch-fidelity
Writes outputs/_comparison/final_image_metrics.json.
"""
import json
import tempfile
from pathlib import Path

import torch_fidelity
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = REPO_ROOT / "outputs"
REAL_DIR = REPO_ROOT / "data" / "zalando-hd-resized" / "test" / "image"
OUT = OUTPUTS / "_comparison" / "final_image_metrics.json"
MODELS = {"catvton": "CatVTON", "ootd": "OOTDiffusion", "idm": "IDM-VTON"}
SIZE = (384, 512)


def resized_copy(paths, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(paths):
        Image.open(p).convert("RGB").resize(SIZE, Image.BICUBIC).save(dst / f"{i:05d}.png")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        resized_copy(sorted(REAL_DIR.glob("*.jpg")), tmp / "real")
        results = {}
        for key, name in MODELS.items():
            runs = sorted(d for d in OUTPUTS.glob(f"{key}_*") if (d / "final.png").exists())
            assert len(runs) == 48, f"{key}: expected 48 runs, found {len(runs)}"
            resized_copy([d / "final.png" for d in runs], tmp / key)
            m = torch_fidelity.calculate_metrics(
                input1=str(tmp / key), input2=str(tmp / "real"), cuda=False,
                fid=True, kid=True, kid_subset_size=48, kid_subsets=1000, verbose=False)
            results[name] = {"fid": m["frechet_inception_distance"],
                             "kid": m["kernel_inception_distance_mean"],
                             "kid_std": m["kernel_inception_distance_std"], "n": len(runs)}
            print(name, results[name])
    OUT.write_text(json.dumps(results, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
