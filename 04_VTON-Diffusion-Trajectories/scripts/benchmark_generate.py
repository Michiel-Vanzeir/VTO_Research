"""
Final images only (no per-step x0-hat) for the standard benchmark scores
(FID, KID, SSIM, LPIPS) of one model on the full VITON-HD test split.

The images come from the same denoising loops as the trajectory runs
(run_with_capture(..., capture=False) in each `*_trajectory.py`), with the
same inputs, seed (42) and guidance scale. The pipeline is loaded once and
then run over every pair, so run it with the model's own env:

    .venv/bin/python      scripts/benchmark_generate.py --model catvton ...
    .venv-ootd/bin/python scripts/benchmark_generate.py --model ootd ...
    .venv-idm/bin/python  scripts/benchmark_generate.py --model idm ...

Settings:
  unpaired  data/.../test_pairs.txt, the standard 2,032 unpaired pairs
  paired    every test person with the garment they wear, so the test photo
            is the ground truth for SSIM / LPIPS

Samplers (--sampler, <name><steps>):
  ddim50    DDIM, eta = 1, 50 steps: the trajectory analysis sampler and
            CatVTON's own default
  unipc30   OOTDiffusion's default scheduler (UniPC, from the checkpoint's
            scheduler config, as in repos/OOTDiffusion/ootd/inference_ootd_hd.py)
  ddpm30    IDM-VTON's default scheduler (DDPM, as in repos/IDM-VTON/inference.py)
The step count is part of the name, so e.g. unipc20 also works.

Output: outputs/_benchmark/<model>/<sampler>/<setting>/<person>__<cloth>.png
at the model's native resolution, plus config.json and timings.csv (seconds
per image) in the same folder. Images that already exist are skipped, so a
killed job just resumes. --shard K/N runs every N-th pair starting at K, for
splitting one setting over several jobs; --limit N runs only the first N
pairs (for testing).

Usage:
    python scripts/benchmark_generate.py --model ootd --sampler unipc30 --setting paired
"""
import argparse
import json
import re
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

DATA = REPO_ROOT / "data" / "zalando-hd-resized"
TEST = DATA / "test"
OUT_ROOT = REPO_ROOT / "outputs" / "_benchmark"
SEED = 42
GUIDANCE = {"catvton": 2.5, "ootd": 2.0, "idm": 2.0}  # Table 1, as in the trajectory runs


def load_pairs(setting: str) -> list[tuple[str, str]]:
    """(person stem, cloth stem), e.g. ('05006_00', '11001_00')."""
    if setting == "unpaired":
        lines = (DATA / "test_pairs.txt").read_text().split("\n")
        return [tuple(Path(f).stem for f in line.split()) for line in lines if line.strip()]
    return [(p.stem, p.stem) for p in sorted((TEST / "image").glob("*.jpg"))]


def parse_sampler(name: str) -> tuple[str, int]:
    m = re.fullmatch(r"(ddim|unipc|ddpm)(\d+)", name)
    if not m:
        raise SystemExit(f"--sampler must look like ddim50 / unipc30 / ddpm30, got {name!r}")
    return m.group(1), int(m.group(2))


def quiet(module):
    """The trajectory scripts draw a tqdm bar per image; one bar per setting is enough here."""
    module.tqdm = partial(tqdm, disable=True)


# ----------------------------------------------------------------- per model
# Each loader returns run(person, cloth) -> final PIL image; the inputs per
# pair are prepared exactly as in that script's main().

def catvton_runner(kind: str, steps: int):
    import catvton_trajectory as m
    if kind != "ddim":
        raise SystemExit("CatVTON is only run with its default DDIM sampler")
    quiet(m)
    pipeline = m.build_pipeline(argparse.Namespace(
        base_model="booksforcharlie/stable-diffusion-inpainting", resume_path="zhengchong/CatVTON",
        attn_version="vitonhd", dtype="bf16"))

    def run(person, cloth, out_dir):
        return m.run_with_capture(
            pipeline,
            Image.open(TEST / "image" / f"{person}.jpg").convert("RGB"),
            Image.open(TEST / "cloth" / f"{cloth}.jpg").convert("RGB"),
            Image.open(TEST / "agnostic-mask-catvton" / f"{person}.png").convert("L"),
            num_inference_steps=steps, guidance_scale=GUIDANCE["catvton"], height=512, width=384,
            seed=SEED, out_dir=out_dir, capture=False)[1]
    return run, type(pipeline.noise_scheduler).__name__


def ootd_runner(kind: str, steps: int):
    import ootd_trajectory as m
    from diffusers import UniPCMultistepScheduler
    from transformers import AutoProcessor, CLIPVisionModelWithProjection
    if kind == "ddpm":
        raise SystemExit("OOTDiffusion runs with ddim (trajectory sampler) or unipc (its default)")
    quiet(m)
    dtype = torch.float16
    pipe = m.build_pipeline(dtype, "cuda")  # DDIM, as in the trajectory runs
    if kind == "unipc":
        # inference_ootd_hd.py: UniPCMultistepScheduler.from_config(<checkpoint scheduler>.config)
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            UniPCMultistepScheduler.load_config(m.MODEL_PATH, subfolder="scheduler"))
    auto_processor = AutoProcessor.from_pretrained(m.VIT_PATH)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(m.VIT_PATH, torch_dtype=dtype).to("cuda")

    def run(person, cloth, out_dir):
        person_image = Image.open(TEST / "image" / f"{person}.jpg").convert("RGB")
        mask, mask_gray = m.build_mask(person_image, TEST / "image-parse-v3" / f"{person}.png",
                                       TEST / "openpose_json" / f"{person}_keypoints.json")
        return m.run_with_capture(
            pipe, image_encoder, auto_processor,
            image_garm=Image.open(TEST / "cloth" / f"{cloth}.jpg").convert("RGB"),
            image_vton=Image.composite(mask_gray, person_image, mask), mask=mask, image_ori=person_image,
            num_inference_steps=steps, image_guidance_scale=GUIDANCE["ootd"], seed=SEED,
            out_dir=out_dir, capture=False)[1]
    return run, type(pipe.scheduler).__name__


def idm_runner(kind: str, steps: int):
    import idmvton_trajectory as m
    from diffusers import DDPMScheduler
    if kind == "unipc":
        raise SystemExit("IDM-VTON runs with ddim (trajectory sampler) or ddpm (its default)")
    quiet(m)
    pipe = m.build_pipeline("cuda")  # DDIM, as in the trajectory runs
    if kind == "ddpm":
        pipe.scheduler = DDPMScheduler.from_pretrained(m.local_model_dir(), subfolder="scheduler")
    pipe.set_progress_bar_config(disable=True)

    def run(person, cloth, out_dir):
        inputs = m.load_inputs(TEST / "image" / f"{person}.jpg", TEST / "cloth" / f"{cloth}.jpg",
                               TEST / "agnostic-mask" / f"{person}_mask.png",
                               TEST / "image-densepose" / f"{person}.jpg")
        return m.run_with_capture(pipe, inputs, m.garment_annotation(f"{cloth}.jpg"), num_inference_steps=steps,
                                  guidance_scale=GUIDANCE["idm"], seed=SEED, out_dir=out_dir,
                                  capture_frames=False)[1]
    return run, type(pipe.scheduler).__name__


RUNNERS = {"catvton": catvton_runner, "ootd": ootd_runner, "idm": idm_runner}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(RUNNERS))
    p.add_argument("--sampler", required=True, help="ddim50, unipc30 (ootd) or ddpm30 (idm)")
    p.add_argument("--setting", required=True, choices=["unpaired", "paired"])
    p.add_argument("--shard", default="0/1", help="K/N: run pairs K, K+N, K+2N, ...")
    p.add_argument("--limit", type=int, default=None, help="only the first N pairs (testing)")
    p.add_argument("--output-root", default=str(OUT_ROOT))
    args = p.parse_args()

    kind, steps = parse_sampler(args.sampler)
    shard, n_shards = (int(x) for x in args.shard.split("/"))
    pairs = load_pairs(args.setting)[:args.limit]
    todo_all = pairs[shard::n_shards]
    out_dir = Path(args.output_root) / args.model / args.sampler / args.setting
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(pe, cl) for pe, cl in todo_all if not (out_dir / f"{pe}__{cl}.png").exists()]
    tag = f"{args.model}/{args.sampler}/{args.setting} shard {shard}/{n_shards}"
    print(f"[bench] {tag}: {len(todo_all)} pairs in shard, {len(todo_all) - len(todo)} done, "
          f"{len(todo)} to do -> {out_dir}", flush=True)
    if not todo:
        return

    t0 = time.time()
    run, scheduler_name = RUNNERS[args.model](kind, steps)
    print(f"[bench] pipeline loaded in {time.time() - t0:.0f} s, scheduler {scheduler_name}", flush=True)
    (out_dir / "config.json").write_text(json.dumps({
        "model": args.model, "sampler": args.sampler, "scheduler": scheduler_name, "steps": steps,
        "eta": 1.0 if kind == "ddim" else None, "setting": args.setting, "seed": SEED,
        "guidance_scale": GUIDANCE[args.model], "n_pairs": len(pairs)}, indent=2))

    timings = out_dir / f"timings_shard{shard}of{n_shards}.csv"
    if not timings.exists():
        timings.write_text("person,cloth,seconds\n")
    spent = []
    for k, (person, cloth) in enumerate(todo, 1):
        # Global RNG too: CatVTON / IDM-VTON VAE-encode their conditioning with
        # latent_dist.sample(); this makes every image independent of the order
        # and of where a resumed job starts.
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        t = time.time()
        image = run(person, cloth, out_dir)
        torch.cuda.synchronize()
        path = out_dir / f"{person}__{cloth}.png"
        tmp = path.with_name(f".{path.name}.tmp.png")  # never leave a half-written PNG behind
        image.save(tmp)
        tmp.rename(path)
        spent.append(time.time() - t)
        with timings.open("a") as f:
            f.write(f"{person},{cloth},{spent[-1]:.3f}\n")
        mean = np.mean(spent[1:] if len(spent) > 1 else spent)  # the first image includes warm-up
        print(f"[bench] {tag}: {k}/{len(todo)} {person} <- {cloth} {spent[-1]:.1f} s "
              f"(mean {mean:.1f} s, ETA {mean * (len(todo) - k) / 60:.0f} min)", flush=True)
    print(f"[bench] {tag}: done, {len(todo)} images in {sum(spent) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
