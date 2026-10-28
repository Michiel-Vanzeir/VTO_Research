"""
Standard final-image scores of the full-test-set runs of benchmark_generate.py.

For every outputs/_benchmark/<model>/<sampler>/<setting>/ folder:
  unpaired  FID, KID
  paired    FID, KID, SSIM, LPIPS (against the test photo of the same person)

One implementation for all models, shared with final_image_metrics.py (the
48-pair FID/KID): every image, real and generated, is first resized to the
common 384x512 frame (bicubic), and FID/KID come from torch-fidelity against
the 2,032 person images of the VITON-HD test split. KID uses 100 subsets of
1,000 images (mean +- std over the subsets).
  SSIM   Wang et al. (2004): Gaussian window sigma = 1.5, per RGB channel,
         averaged over the image (skimage; same definition as torchmetrics).
  LPIPS  Zhang et al. (2018), AlexNet features (the `lpips` package default).
SSIM and LPIPS are over the whole image, not only the garment region.

Per folder the result is cached in metrics.json and only recomputed when the
number of images changes (or with --force). Folders that are not complete
(fewer images than pairs) are skipped unless --allow-partial is given, which
the 5-pair test uses. Writes outputs/_benchmark/metrics.json and prints the
rows of the benchmark table.

Usage:
    python scripts/benchmark_metrics.py                  # needs torch-fidelity and lpips
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_image_metrics import REAL_DIR, SIZE, fid_kid, resized_copy  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH = REPO_ROOT / "outputs" / "_benchmark"
REAL_RESIZED = BENCH / f"_real_{SIZE[0]}x{SIZE[1]}"
N_TEST = 2032
KID_SUBSET_SIZE, KID_SUBSETS = 1000, 100
MODELS = {"catvton": "CatVTON", "ootd": "OOTDiffusion", "idm": "IDM-VTON"}
SAMPLERS = {"ddim50": "DDIM 50", "unipc30": "UniPC 30", "ddpm30": "DDPM 30"}


def real_images() -> Path:
    """The 2,032 test person images at SIZE, named <stem>.png; made once."""
    reals = sorted(REAL_DIR.glob("*.jpg"))
    if len(list(REAL_RESIZED.glob("*.png"))) != len(reals):
        print(f"[metrics] resizing {len(reals)} real images -> {REAL_RESIZED}", flush=True)
        resized_copy(reals, REAL_RESIZED, names=[p.stem for p in reals])
    return REAL_RESIZED


def paired_scores(gen_dir: Path, real_dir: Path, device) -> dict:
    import lpips
    net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    ssim, lp = [], []
    for p in sorted(gen_dir.glob("*.png")):
        gen = np.asarray(Image.open(p).convert("RGB"))
        gt = np.asarray(Image.open(real_dir / p.name).convert("RGB"))
        ssim.append(structural_similarity(gt, gen, channel_axis=2, data_range=255, gaussian_weights=True,
                                          sigma=1.5, use_sample_covariance=False))
        to_t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(device)  # noqa: E731
        with torch.no_grad():
            lp.append(net(to_t(gt), to_t(gen)).item())
    return {"ssim": float(np.mean(ssim)), "lpips": float(np.mean(lp))}


def score_folder(folder: Path, real_dir: Path, cuda: bool) -> dict:
    gens = sorted(p for p in folder.glob("*.png") if not p.name.startswith("."))
    setting = folder.name
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # <person>__<cloth>.png -> <person>.png: every person occurs once per setting,
        # and in the paired setting that is the name of its ground truth
        resized_copy(gens, tmp, names=[p.stem.split("__")[0] for p in gens])
        # no torch-fidelity feature cache: 0.3.0 reads it back with torch.load,
        # which refuses its numpy arrays from torch 2.6 on (weights_only=True)
        res = {"n": len(gens), **fid_kid(
            tmp, real_dir, kid_subset_size=min(KID_SUBSET_SIZE, len(gens)), kid_subsets=KID_SUBSETS, cuda=cuda)}
        if setting == "paired":
            res.update(paired_scores(tmp, real_dir, "cuda" if cuda else "cpu"))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--allow-partial", action="store_true", help="also score folders with < 2,032 images")
    p.add_argument("--force", action="store_true", help="recompute even if metrics.json is up to date")
    p.add_argument("--root", default=str(BENCH))
    args = p.parse_args()

    root = Path(args.root)
    cuda = torch.cuda.is_available()
    real_dir = real_images()
    results = {}
    for folder in sorted(root.glob("*/*/*paired")):
        key = "/".join(folder.parts[-3:])
        n = len([q for q in folder.glob("*.png") if not q.name.startswith(".")])
        if n < N_TEST and not args.allow_partial:
            print(f"[metrics] skip {key}: {n}/{N_TEST} images (--allow-partial to score anyway)")
            continue
        cached = folder / "metrics.json"
        if cached.exists() and not args.force and json.loads(cached.read_text()).get("n") == n:
            results[key] = json.loads(cached.read_text())
        else:
            print(f"[metrics] scoring {key} ({n} images)", flush=True)
            results[key] = score_folder(folder, real_dir, cuda)
            cached.write_text(json.dumps(results[key], indent=2))
        print(f"[metrics] {key}: {results[key]}", flush=True)
    (root / "metrics.json").write_text(json.dumps(results, indent=2))
    print("wrote", root / "metrics.json")

    # rows of the benchmark table: FID_u, KID_u, FID_p, KID_p, SSIM, LPIPS
    fmt_kid = lambda r: f"${1e3 * r['kid']:.2f} \\pm {1e3 * r['kid_std']:.2f}$"  # noqa: E731
    for model, name in MODELS.items():
        for sampler, sname in SAMPLERS.items():
            u, pr = results.get(f"{model}/{sampler}/unpaired"), results.get(f"{model}/{sampler}/paired")
            if not (u or pr):
                continue
            cells = [f"{u['fid']:.2f}", fmt_kid(u)] if u else ["--", "--"]
            cells += [f"{pr['fid']:.2f}", fmt_kid(pr), f"{pr['ssim']:.3f}", f"{pr['lpips']:.3f}"] if pr else ["--"] * 4
            print(f"  {name} & {sname} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
