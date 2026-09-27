"""
Run IDM-VTON (SDXL-inpainting try-on, VITON-HD checkpoint) inference while
capturing, at every denoising step, both:
  - the model's running x0-prediction (decoded VAE estimate of the clean image)
  - the current noisy latent z_t (decoded the same way)

Same output layout as catvton_trajectory.py / ootd_trajectory.py, so
metrics.py / viz.py / compare_trajectories.py work on it unchanged.

Unlike those two scripts, this one does NOT reimplement the pipeline's
denoising loop: IDM-VTON's TryonPipeline.__call__ (repos/IDM-VTON/src/
tryon_pipeline.py) is ~400 lines of SDXL conditioning (added time ids,
IP-adapter image embeds, per-step GarmentNet reference features, pose
latents) that would be easy to get subtly wrong by hand. Instead the
scheduler's `step()` is wrapped: the pipeline calls it once per step with
exactly (noise_pred after CFG, t, z_t), which is everything needed for
x0-hat, so capture happens there while the loop itself runs untouched. The
wrapper keeps `eta`/`generator` in its signature because the pipeline's
prepare_extra_step_kwargs inspects that signature to decide what to pass.

Inputs follow upstream inference.py's VITON-HD test path exactly:
  - person image resized to 768x1024, `agnostic-mask/<stem>_mask.png` as
    the inpainting mask, `image-densepose/<stem>.jpg` as the pose input;
  - captions built from repos/IDM-VTON/vitonhd_test_tagged.json the same
    way its VitonHdTestDataset does ("model is wearing a <sleeve> <neck>
    <item>", "a photo of ..."), falling back to "shirts" like upstream;
  - fp16 weights under torch.cuda.amp.autocast, guidance 2.0.

Deliberate deviations, same reasoning as ootd_trajectory.py:
  - DDPMScheduler (upstream default, 30 steps) is swapped for DDIMScheduler
    with eta=1.0 and 50 steps from the same config, matching the CatVTON
    and OOTDiffusion runs' step schedule. alphas_cumprod is unchanged.
  - The VAE is kept in fp32 and our per-step decodes run with autocast
    disabled. IDM-VTON's VAE is an fp16-safe SDXL VAE (force_upcast=false,
    so the pipeline never re-casts it), but the high-noise x0-hat latents
    we decode are far outside the range it was made fp16-safe for, and
    fp32 costs little on the 24GB cards this model needs anyway.

Usage:
    python idmvton_trajectory.py \
        --person data/.../test/image/xxx.jpg \
        --cloth  data/.../test/cloth/yyy.jpg \
        --run-name idm_pair1_solid \
        --steps 50 --guidance-scale 2.0 --seed 42
"""
import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
IDM_REPO = REPO_ROOT / "repos" / "IDM-VTON"
sys.path.insert(0, str(IDM_REPO))
sys.path.insert(0, str(SCRIPT_DIR))

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline  # noqa: E402
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref  # noqa: E402
from src.unet_hacked_tryon import UNet2DConditionModel  # noqa: E402

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from torchvision import transforms  # noqa: E402
from transformers import (  # noqa: E402
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from common import add_fork_args, decode_latents, fork_generators, run_forks, to_pil, verify_pred_original_sample  # noqa: E402

MODEL_ID = "yisol/IDM-VTON"
HEIGHT, WIDTH = 1024, 768
NEGATIVE_PROMPT = "monochrome, lowres, bad anatomy, worst quality, low quality"
TAGGED_JSON = IDM_REPO / "vitonhd_test_tagged.json"


def garment_annotation(cloth_name: str) -> str:
    """Upstream VitonHdTestDataset's caption fragment for a cloth file."""
    if TAGGED_JSON.exists():
        data = json.loads(TAGGED_JSON.read_text())
        for entries in data.values():
            for elem in entries:
                if elem["file_name"] != cloth_name:
                    continue
                parts = []
                for template in ("sleeveLength", "neckLine", "item"):
                    for tag in elem["tag_info"]:
                        if tag["tag_name"] == template and tag["tag_category"] is not None:
                            parts.append(tag["tag_category"] + " ")
                return "".join(parts)
    return "shirts"


class CaptureStep:
    """Replaces scheduler.step: runs the real step, then decodes and saves
    z_t and x0-hat for that step. capture=False (fork mode) skips the
    decoding; from step `fork_step` on, the step's noise comes from
    `fork_gen` instead of the pipeline's generator (see common.py)."""

    def __init__(self, scheduler, vae, out_dir: Path, total_steps: int, capture: bool = True,
                 fork_step=None, fork_gen=None):
        self.scheduler = scheduler
        # the CLASS method, not scheduler.step: in fork mode the pipeline runs
        # many times and an instance attribute would be the previous wrapper
        self.orig_step = type(scheduler).step.__get__(scheduler, type(scheduler))
        self.vae = vae
        self.capture = capture
        self.fork_step, self.fork_gen = fork_step, fork_gen
        self.frames_x0_dir = out_dir / "frames_x0"
        self.frames_zt_dir = out_dir / "frames_zt"
        if capture:
            self.frames_x0_dir.mkdir(parents=True, exist_ok=True)
            self.frames_zt_dir.mkdir(parents=True, exist_ok=True)
        self.timesteps = []
        self.final_latents = None
        self.pbar = tqdm(total=total_steps, desc=out_dir.name)

    def __call__(self, model_output, timestep, sample, eta=0.0, use_clipped_model_output=False,
                 generator=None, variance_noise=None, return_dict=True):
        i = len(self.timesteps)
        if self.fork_gen is not None and i >= self.fork_step:
            generator = self.fork_gen
        out = self.orig_step(model_output, timestep, sample, eta=eta,
                             use_clipped_model_output=use_clipped_model_output,
                             generator=generator, variance_noise=variance_noise, return_dict=True)
        self.timesteps.append(int(timestep))
        self.final_latents = out.prev_sample
        self.pbar.update()
        if not self.capture:
            return out if return_dict else (out.prev_sample,)

        if i == 0:
            x0_hat, _ = verify_pred_original_sample(self.scheduler, out, model_output, timestep, sample)
        else:
            x0_hat = out.pred_original_sample

        with torch.autocast("cuda", enabled=False):
            decoded = decode_latents(self.vae, torch.cat([sample, x0_hat], dim=0), torch.float32)
        zt_pil, x0_pil = to_pil(decoded[0:1])[0], to_pil(decoded[1:2])[0]
        zt_pil.save(self.frames_zt_dir / f"step_{i:03d}.png")
        x0_pil.save(self.frames_x0_dir / f"step_{i:03d}.png")
        return out if return_dict else (out.prev_sample,)


def local_model_dir() -> str:
    """Local snapshot folder of MODEL_ID in the HF cache (fetched by
    cluster/setup.sh). Everything loads from this path rather than the repo
    id: diffusers 0.25's DiffusionPipeline.from_pretrained calls the Hub API
    (model_info) for a repo id even when HF_HUB_OFFLINE=1 is set, which
    crashes on offline condor execute nodes. A local path never hits the Hub."""
    try:
        return snapshot_download(MODEL_ID, local_files_only=True)
    except Exception:
        return snapshot_download(MODEL_ID)  # not cached yet and online: fetch it


def build_pipeline(device: str) -> TryonPipeline:
    dtype = torch.float16
    model_dir = local_model_dir()
    print(f"[idm] loading weights from {model_dir}")
    unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder="unet", torch_dtype=dtype)
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(model_dir, subfolder="unet_encoder", torch_dtype=dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_dir, subfolder="image_encoder", torch_dtype=dtype)
    text_encoder_one = CLIPTextModel.from_pretrained(model_dir, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(model_dir, subfolder="text_encoder_2", torch_dtype=dtype)
    tokenizer_one = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer_2", use_fast=False)
    vae = AutoencoderKL.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch.float32)
    ddpm = DDPMScheduler.from_pretrained(model_dir, subfolder="scheduler")
    scheduler = DDIMScheduler.from_config(ddpm.config)

    for m in (unet, unet_encoder, image_encoder, text_encoder_one, text_encoder_two, vae):
        m.requires_grad_(False)
        m.eval()

    pipe = TryonPipeline.from_pretrained(
        model_dir,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=scheduler,
        image_encoder=image_encoder,
        unet_encoder=unet_encoder,
        torch_dtype=dtype,
    ).to(device)
    # from_pretrained's torch_dtype would otherwise have cast our fp32 VAE back down
    pipe.vae.to(torch.float32)
    # decode z_t and x0-hat one at a time; same values, half the peak activation memory
    pipe.vae.enable_slicing()
    return pipe


def load_inputs(person_path: Path, cloth_path: Path, mask_path: Path, densepose_path: Path):
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    person = Image.open(person_path).convert("RGB").resize((WIDTH, HEIGHT))
    cloth = Image.open(cloth_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").resize((WIDTH, HEIGHT))
    densepose = Image.open(densepose_path).convert("RGB").resize((WIDTH, HEIGHT))
    return {
        "image": ((normalize(person) + 1.0) / 2.0).unsqueeze(0),  # [0,1], as inference.py passes it
        "inpaint_mask": transforms.ToTensor()(mask)[:1].unsqueeze(0),
        "pose_img": normalize(densepose).unsqueeze(0),
        "cloth_pure": normalize(cloth).unsqueeze(0),
        "cloth_clip": CLIPImageProcessor()(images=cloth, return_tensors="pt").pixel_values,
        "mask_pil": mask,
    }


@torch.no_grad()
def run_with_capture(pipe, inputs, annotation: str, num_inference_steps: int, guidance_scale: float,
                     seed: int, out_dir: Path, capture_frames: bool = True, fork_step=None, fork_seed=None):
    """Returns (timesteps, final image). capture_frames=False (fork mode)
    skips all per-step decoding/saving; fork_step/fork_seed: see common.py."""
    device = pipe.device
    generator, fork_gen = fork_generators(device, seed, fork_step, fork_seed)
    capture = CaptureStep(pipe.scheduler, pipe.vae, out_dir, num_inference_steps, capture=capture_frames,
                          fork_step=fork_step, fork_gen=fork_gen)
    pipe.scheduler.step = capture
    assert "eta" in inspect.signature(pipe.scheduler.step).parameters

    with torch.cuda.amp.autocast(), torch.inference_mode():
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
            ["model is wearing a " + annotation],
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=[NEGATIVE_PROMPT],
        )
        prompt_embeds_c, _, _, _ = pipe.encode_prompt(
            ["a photo of " + annotation],
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=[NEGATIVE_PROMPT],
        )
        images = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            eta=1.0,
            strength=1.0,
            pose_img=inputs["pose_img"],
            text_embeds_cloth=prompt_embeds_c,
            cloth=inputs["cloth_pure"].to(device),
            mask_image=inputs["inpaint_mask"],
            image=inputs["image"],
            height=HEIGHT,
            width=WIDTH,
            guidance_scale=guidance_scale,
            ip_adapter_image=inputs["cloth_clip"],
        )[0]
    capture.pbar.close()

    # IDM-VTON has no RePaint-style blend (its UNet takes 13 input channels,
    # not the 4-channel path that blends), so the last stepped latent IS the
    # pipeline's output latent. Decode it the same way as every frame for
    # final.png, and keep the pipeline's own postprocessed output alongside
    # as a sanity check that the wrapped loop produced the normal result.
    with torch.autocast("cuda", enabled=False):
        final_pil = to_pil(decode_latents(pipe.vae, capture.final_latents, torch.float32))[0]
    if capture_frames:
        final_pil.save(out_dir / "final.png")
        images[0].save(out_dir / "final_pipeline.png")
    return capture.timesteps, final_pil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--person", required=True)
    p.add_argument("--cloth", required=True)
    p.add_argument("--mask", default=None, help="defaults to <person's dataset>/agnostic-mask/<stem>_mask.png")
    p.add_argument("--densepose", default=None, help="defaults to <person's dataset>/image-densepose/<stem>.jpg")
    p.add_argument("--run-name", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", default=str(REPO_ROOT / "outputs"))
    p.add_argument("--device", default="cuda")
    add_fork_args(p)
    args = p.parse_args()

    person_path, cloth_path = Path(args.person), Path(args.cloth)
    dataset_root = person_path.parents[1]  # .../test/image/xxx.jpg -> .../test
    mask_path = Path(args.mask) if args.mask else dataset_root / "agnostic-mask" / f"{person_path.stem}_mask.png"
    densepose_path = Path(args.densepose) if args.densepose else dataset_root / "image-densepose" / f"{person_path.stem}.jpg"

    out_dir = Path(args.output_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    inputs = load_inputs(person_path, cloth_path, mask_path, densepose_path)
    annotation = garment_annotation(cloth_path.name)
    print(f"[idm] garment annotation: {annotation!r}")
    pipe = build_pipeline(args.device)

    if args.fork_steps:
        run_one = lambda k, s: run_with_capture(  # noqa: E731
            pipe, inputs, annotation, num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
            seed=args.seed, out_dir=out_dir, capture_frames=False, fork_step=k, fork_seed=s)[1]
        run_forks(args, out_dir, run_one, {"model": "IDM-VTON", "num_inference_steps": args.steps, "inputs": {
            "person": str(person_path.resolve()), "cloth": str(cloth_path.resolve()), "mask": str(mask_path.resolve())}})
        return

    inputs["mask_pil"].save(out_dir / "mask.png")
    timesteps, _ = run_with_capture(
        pipe, inputs, annotation,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        out_dir=out_dir,
    )

    config = {
        "model": "IDM-VTON",
        "model_id": MODEL_ID,
        "scheduler": type(pipe.scheduler).__name__,
        "eta": 1.0,
        "weight_dtype": "fp16 (vae fp32)",
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": HEIGHT,
        "width": WIDTH,
        "caption": "model is wearing a " + annotation,
        "inputs": {
            "person": str(person_path.resolve()),
            "cloth": str(cloth_path.resolve()),
            "mask": str(mask_path.resolve()),
            "densepose": str(densepose_path.resolve()),
        },
        "timesteps": timesteps,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Done. Wrote {out_dir}")


if __name__ == "__main__":
    main()
