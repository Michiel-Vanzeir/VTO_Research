"""
Run CatVTON inference while capturing, at every denoising step, both:
  - the model's running x0-prediction (decoded VAE estimate of the clean image)
  - the current noisy latent z_t (decoded the same way)

CatVTON is not a stock diffusers pipeline (its UNet takes a 9-channel
inpainting input and there's no `callback_on_step_end` hook), so this
reimplements CatVTONPipeline.__call__ (repos/CatVTON/model/pipeline.py)
step-by-step instead of calling it as a black box.

Usage:
    python catvton_trajectory.py \
        --person data/.../image/xxx.jpg \
        --cloth  data/.../cloth/yyy.jpg \
        --mask   data/.../agnostic-mask/xxx_mask.png \
        --run-name pair1_solid \
        --steps 50 --guidance-scale 2.5 --seed 42
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CATVTON_REPO = REPO_ROOT / "repos" / "CatVTON"
sys.path.insert(0, str(CATVTON_REPO))

from model.pipeline import CatVTONPipeline  # noqa: E402
from utils import (  # noqa: E402
    compute_vae_encodings,
    prepare_image,
    prepare_mask_image,
)
from diffusers.utils.torch_utils import randn_tensor  # noqa: E402

sys.path.insert(0, str(SCRIPT_DIR))
from common import (  # noqa: E402
    add_fork_args, decode_latents, fork_generators, run_forks, step_kwargs, to_pil, verify_pred_original_sample,
)

# --- Empirically confirmed spatial-concat layout ---------------------------
# CatVTON's pipeline.py does (concat_dim = -2, i.e. the LATENT HEIGHT axis):
#     masked_latent_concat = cat([masked_latent(person), condition_latent(garment)], dim=-2)
#     ... final decode: latents.split(H // 2, dim=-2)[0]   <- FIRST half
# so the person occupies the TOP half after concatenation, garment the bottom
# half. This is re-verified visually per-run in `axis_check.png` (see
# `save_axis_check` below) rather than just trusted from reading the source.
CONCAT_DIM = -2
PERSON_HALF = 0  # index into .chunk(2, dim=CONCAT_DIM)


def slice_person(latents: torch.Tensor) -> torch.Tensor:
    return latents.chunk(2, dim=CONCAT_DIM)[PERSON_HALF]


def slice_garment(latents: torch.Tensor) -> torch.Tensor:
    return latents.chunk(2, dim=CONCAT_DIM)[1 - PERSON_HALF]


def save_axis_check(vae, latents: torch.Tensor, weight_dtype, out_path: Path):
    """Decode BOTH halves of the final latent side-by-side and save, so the
    person/garment half assignment above can be checked by eye rather than
    only assumed from reading pipeline.py."""
    person = slice_person(latents)
    garment = slice_garment(latents)
    decoded = decode_latents(vae, torch.cat([person, garment], dim=0), weight_dtype)
    person_img, garment_img = to_pil(decoded[0:1])[0], to_pil(decoded[1:2])[0]
    w, h = person_img.size
    combo = Image.new("RGB", (w * 2 + 20, h), (30, 30, 30))
    combo.paste(person_img, (0, 0))
    combo.paste(garment_img, (w + 20, 0))
    combo.save(out_path)
    print(f"[axis-check] saved {out_path} — left=PERSON_HALF({PERSON_HALF}), right=other half. "
          f"Confirm left looks like a person, right like a garment.")


@torch.no_grad()
def run_with_capture(
    pipeline: CatVTONPipeline,
    person_image: Image.Image,
    cloth_image: Image.Image,
    mask_image: Image.Image,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: int,
    out_dir: Path,
    capture: bool = True,
    fork_step=None,
    fork_seed=None,
):
    """Returns (timesteps, final person image). capture=False (fork mode)
    skips all per-step decoding/saving; fork_step/fork_seed: see common.py."""
    frames_x0_dir = out_dir / "frames_x0"
    frames_zt_dir = out_dir / "frames_zt"
    if capture:
        frames_x0_dir.mkdir(parents=True, exist_ok=True)
        frames_zt_dir.mkdir(parents=True, exist_ok=True)

    device = pipeline.device
    weight_dtype = pipeline.weight_dtype
    vae, unet, scheduler = pipeline.vae, pipeline.unet, pipeline.noise_scheduler

    generator, fork_gen = fork_generators(device, seed, fork_step, fork_seed)

    image, condition_image, mask = pipeline.check_inputs(person_image, cloth_image, mask_image, width, height)
    image = prepare_image(image).to(device, dtype=weight_dtype)
    condition_image = prepare_image(condition_image).to(device, dtype=weight_dtype)
    mask = prepare_mask_image(mask).to(device, dtype=weight_dtype)

    masked_image = image * (mask < 0.5)
    masked_latent = compute_vae_encodings(masked_image, vae)
    condition_latent = compute_vae_encodings(condition_image, vae)
    mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
    del image, mask, condition_image

    masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=CONCAT_DIM)
    mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=CONCAT_DIM)

    latents = randn_tensor(
        masked_latent_concat.shape, generator=generator, device=masked_latent_concat.device, dtype=weight_dtype,
    )
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    latents = latents * scheduler.init_noise_sigma

    do_cfg = guidance_scale > 1.0
    if do_cfg:
        masked_latent_concat = torch.cat([
            torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=CONCAT_DIM),
            masked_latent_concat,
        ])
        mask_latent_concat = torch.cat([mask_latent_concat] * 2)

    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta=1.0)
    step_timesteps = []

    for i, t in enumerate(tqdm(timesteps, desc=out_dir.name)):
        z_t_person = slice_person(latents)  # z_t going INTO this step

        model_input = torch.cat([latents] * 2) if do_cfg else latents
        model_input = scheduler.scale_model_input(model_input, t)
        unet_input = torch.cat([model_input, mask_latent_concat, masked_latent_concat], dim=1)

        noise_pred = unet(unet_input, t.to(device), encoder_hidden_states=None, return_dict=False)[0]
        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        step_output = scheduler.step(noise_pred, t, latents, **step_kwargs(i, extra_step_kwargs, fork_step, fork_gen))
        step_timesteps.append(int(t))
        if not capture:
            latents = step_output.prev_sample
            continue

        if i == 0:
            x0_hat, _ = verify_pred_original_sample(scheduler, step_output, noise_pred, t, latents)
        else:
            x0_hat = getattr(step_output, "pred_original_sample", None)
            if x0_hat is None:
                alpha_bar_t = scheduler.alphas_cumprod.to(latents.device)[t]
                x0_hat = (latents - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
        x0_hat_person = slice_person(x0_hat)

        # one batched VAE decode per step instead of two separate calls
        batch = torch.cat([z_t_person, x0_hat_person], dim=0)
        decoded = decode_latents(vae, batch, weight_dtype)
        zt_pil, x0_pil = to_pil(decoded[0:1])[0], to_pil(decoded[1:2])[0]
        zt_pil.save(frames_zt_dir / f"step_{i:03d}.png")
        x0_pil.save(frames_x0_dir / f"step_{i:03d}.png")

        latents = step_output.prev_sample

    final_person = slice_person(latents)
    final_pil = to_pil(decode_latents(vae, final_person, weight_dtype))[0]
    if capture:
        save_axis_check(vae, latents, weight_dtype, out_dir / "axis_check.png")
        final_pil.save(out_dir / "final.png")

    return step_timesteps, final_pil


def build_pipeline(args, device="cuda"):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    pipeline = CatVTONPipeline(
        base_ckpt=args.base_model,
        attn_ckpt=args.resume_path,
        attn_ckpt_version=args.attn_version,
        weight_dtype=dtype,
        device=device,
        skip_safety_check=True,
    )
    return pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--person", required=True)
    p.add_argument("--cloth", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--base-model", default="booksforcharlie/stable-diffusion-inpainting")
    p.add_argument("--resume-path", default="zhengchong/CatVTON")
    p.add_argument("--attn-version", default="vitonhd", choices=["mix", "vitonhd", "dresscode"])
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--output-root", default=str(REPO_ROOT / "outputs"))
    add_fork_args(p)
    args = p.parse_args()

    out_dir = Path(args.output_root) / args.run_name
    person_image = Image.open(args.person).convert("RGB")
    cloth_image = Image.open(args.cloth).convert("RGB")
    mask_image = Image.open(args.mask).convert("L")
    pipeline = build_pipeline(args)
    inputs = {"person": str(Path(args.person).resolve()), "cloth": str(Path(args.cloth).resolve()),
              "mask": str(Path(args.mask).resolve())}

    if args.fork_steps:
        run_one = lambda k, s: run_with_capture(  # noqa: E731
            pipeline, person_image, cloth_image, mask_image, num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale, height=args.height, width=args.width, seed=args.seed,
            out_dir=out_dir, capture=False, fork_step=k, fork_seed=s)[1]
        run_forks(args, out_dir, run_one, {"model": "CatVTON", "num_inference_steps": args.steps, "inputs": inputs})
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    timesteps, _ = run_with_capture(
        pipeline, person_image, cloth_image, mask_image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
        out_dir=out_dir,
    )

    config = {
        "model": "CatVTON",
        "base_model_path": args.base_model,
        "resume_path": args.resume_path,
        "attn_ckpt_version": args.attn_version,
        "scheduler": type(pipeline.noise_scheduler).__name__,
        "eta": 1.0,
        "weight_dtype": args.dtype,
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "concat_dim": CONCAT_DIM,
        "person_half_index": PERSON_HALF,
        "inputs": {
            "person": str(Path(args.person).resolve()),
            "cloth": str(Path(args.cloth).resolve()),
            "mask": str(Path(args.mask).resolve()),
        },
        "timesteps": timesteps,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Done. Wrote {out_dir}")


if __name__ == "__main__":
    main()
