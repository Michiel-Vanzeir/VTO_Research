"""
Run OOTDiffusion (HD, upper-body) inference while capturing, at every
denoising step, both:
  - the model's running x0-prediction (decoded VAE estimate of the clean image)
  - the current noisy latent z_t (decoded the same way)

Mirrors catvton_trajectory.py's approach and output layout so the two
models' trajectories are directly comparable with the same metrics/viz
scripts. Like that script, this reimplements OotdPipeline.__call__
(repos/OOTDiffusion/ootd/pipelines_ootd/pipeline_ootd.py) step-by-step
instead of calling it as a black box, because the noise-prediction UNet
call needs to sit between our capture points. Unlike CatVTON, OOTDiffusion
*does* support `callback_on_step_end`, but that callback only exposes the
already-stepped `latents` -- not the noise prediction needed to compute
x0-hat -- so a manual loop is still required to get x0-hat every step.

Two deliberate deviations from the upstream demo (repos/OOTDiffusion/run/run_ootd.py),
both to keep this run comparable to the CatVTON runs in this project:
  - Preprocessing (agnostic mask) is derived from the VITON-HD dataset's own
    precomputed human-parsing (`image-parse-v3`) and OpenPose keypoints
    (`openpose_json`) instead of running OOTDiffusion's bundled parsing/
    openpose models live. Those precomputed annotations were produced at
    384x512 in the original VITON-HD pipeline convention that
    `get_mask_location` (utils_ootd.py) expects; ours are stored at full
    768x1024 resolution, so we downscale keypoints by 0.5 (768/1024 -> 384/512)
    before calling it and let the function itself NEAREST-downsize the
    parse map -- reproducing exactly what run_ootd.py does when it runs
    those models on a 384x512-resized person image.
  - The scheduler is swapped from UniPCMultistepScheduler (OOTDiffusion's
    default) to DDIMScheduler with the same step count/eta as the CatVTON
    runs, so both models are sampled with the same step schedule for the
    trajectory comparison. The UNet is epsilon-prediction and schedule-
    agnostic at inference; alphas_cumprod is a property of training, not of
    which sampler reads it, so this does not require retraining or change
    what the UNet predicts at a given t.

Usage:
    python ootd_trajectory.py \
        --person data/.../image/xxx.jpg \
        --cloth  data/.../cloth/yyy.jpg \
        --run-name pair1_solid \
        --steps 50 --image-guidance-scale 2.0 --seed 42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OOTD_REPO = REPO_ROOT / "repos" / "OOTDiffusion"
sys.path.insert(0, str(OOTD_REPO))
sys.path.insert(0, str(SCRIPT_DIR))
# checkpoints/ootd/model_index.json records module paths as bare
# "pipelines_ootd.*" (upstream's own inference scripts put ootd/ itself on
# sys.path, see inference_ootd_hd.py), so from_pretrained's dynamic class
# lookup needs that too, in addition to the "ootd.pipelines_ootd.*" import
# path used directly above.
sys.path.insert(0, str(OOTD_REPO / "ootd"))

from ootd.pipelines_ootd.pipeline_ootd import OotdPipeline  # noqa: E402
from ootd.pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel  # noqa: E402
from ootd.pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel  # noqa: E402
from run.utils_ootd import get_mask_location  # noqa: E402

from diffusers import AutoencoderKL, DDIMScheduler  # noqa: E402
from transformers import AutoProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection  # noqa: E402

from common import decode_latents, to_pil, verify_pred_original_sample  # noqa: E402

CHECKPOINTS = OOTD_REPO / "checkpoints"
VIT_PATH = CHECKPOINTS / "clip-vit-large-patch14"
VAE_PATH = CHECKPOINTS / "ootd"
UNET_PATH = CHECKPOINTS / "ootd" / "ootd_hd" / "checkpoint-36000"
MODEL_PATH = CHECKPOINTS / "ootd"

CATEGORY = "upper_body"  # HD checkpoint only supports upper-body (run_ootd.py enforces category==0)
MODEL_TYPE = "hd"


def load_keypoints_384x512(json_path: Path, orig_size=(768, 1024)) -> dict:
    """Load VITON-HD's precomputed OpenPose json (25 or 18 keypoints as
    x,y,confidence triples at full image resolution) and reshape it into
    the flat (x, y) x 18 list at 384x512 that get_mask_location expects
    (see module docstring). BODY_25 and BODY_18 share the same ordering
    for the first 8 points (nose, neck, shoulders, elbows, wrists) that
    get_mask_location actually indexes into."""
    data = json.loads(json_path.read_text())
    people = data.get("people", [])
    if not people:
        raise ValueError(f"no people detected in {json_path}")
    raw = np.array(people[0]["pose_keypoints_2d"], dtype=np.float32).reshape(-1, 3)[:, :2]
    sx, sy = 384.0 / orig_size[0], 512.0 / orig_size[1]
    raw[:, 0] *= sx
    raw[:, 1] *= sy
    return {"pose_keypoints_2d": raw[:18].tolist()}


# VITON-HD's image-parse-v3 maps use the dataset's own LIP-style 20-class
# label scheme, NOT the ATR-style 18-class scheme get_mask_location()/
# label_map (utils_ootd.py) expects. Confirmed two ways: (1) direct pixel
# inspection -- in this scheme, label 5 is spatially exactly the visible
# shirt, not label 4; (2) OOTDiffusion's own live-parsing post-processing
# (parsing_api.py's delete_irregular: `parsing_result == 4` for upper_cloth,
# `== 7` for dresses) independently confirms 4/7 is the scheme
# get_mask_location's callers actually expect. Without this remap,
# get_mask_location's `parse_array == 4/7` garment lookup matches almost
# nothing, and the only surviving mask content is the OpenPose-keypoint-
# driven arm sliver (labels 14/15 happen to mean "left/right arm" in BOTH
# schemes, which is why the broken mask was arm-shaped, not empty).
_LIP_TO_ATR = {
    0: 0, 1: 1, 2: 2, 3: 0, 4: 3, 5: 4, 6: 7, 7: 4, 8: 0, 9: 6,
    10: 0, 11: 17, 12: 5, 13: 11, 14: 14, 15: 15, 16: 12, 17: 13, 18: 9, 19: 10,
}
_LIP_TO_ATR_LUT = np.zeros(256, dtype=np.uint8)
for _lip_val, _atr_val in _LIP_TO_ATR.items():
    _LIP_TO_ATR_LUT[_lip_val] = _atr_val


def remap_lip_to_atr_parse(model_parse: Image.Image) -> Image.Image:
    return Image.fromarray(_LIP_TO_ATR_LUT[np.array(model_parse)], mode="L")


def build_mask(person_image: Image.Image, parse_path: Path, keypoints_path: Path) -> tuple[Image.Image, Image.Image]:
    model_parse = Image.open(parse_path)  # LIP-scheme label map (VITON-HD's own convention)
    model_parse = remap_lip_to_atr_parse(model_parse)  # -> ATR scheme, what get_mask_location expects
    keypoints = load_keypoints_384x512(keypoints_path, orig_size=person_image.size)
    mask, mask_gray = get_mask_location(MODEL_TYPE, CATEGORY, model_parse, keypoints)
    mask = mask.resize(person_image.size, Image.NEAREST)
    mask_gray = mask_gray.resize(person_image.size, Image.NEAREST)
    return mask, mask_gray


def build_pipeline(dtype: torch.dtype, device: str) -> OotdPipeline:
    # The levihsu/OOTDiffusion mirror ships plain (non-"fp16"-variant)
    # checkpoints -- vae/text_encoder as fp32 .bin, unet_garm/unet_vton as
    # fp32 .safetensors -- unlike upstream's own inference script, which
    # assumes a variant="fp16" layout that doesn't exist in this download.
    # torch_dtype still downcasts everything to `dtype` at load time.
    vae = AutoencoderKL.from_pretrained(VAE_PATH, subfolder="vae", torch_dtype=dtype)
    unet_garm = UNetGarm2DConditionModel.from_pretrained(
        UNET_PATH, subfolder="unet_garm", torch_dtype=dtype, use_safetensors=True,
    )
    unet_vton = UNetVton2DConditionModel.from_pretrained(
        UNET_PATH, subfolder="unet_vton", torch_dtype=dtype, use_safetensors=True,
    )
    pipe = OotdPipeline.from_pretrained(
        MODEL_PATH,
        unet_garm=unet_garm,
        unet_vton=unet_vton,
        vae=vae,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # 768x1024 is 3x the pixel count of CatVTON's 512x384 run; slicing/tiling
    # the VAE decode (which we call every step, for both z_t and x0-hat) is a
    # free memory saving on an 8GB card with no effect on the decoded values.
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


@torch.no_grad()
def run_with_capture(
    pipe: OotdPipeline,
    image_encoder,
    auto_processor,
    image_garm: Image.Image,
    image_vton: Image.Image,
    mask: Image.Image,
    image_ori: Image.Image,
    num_inference_steps: int,
    image_guidance_scale: float,
    seed: int,
    out_dir: Path,
):
    frames_x0_dir = out_dir / "frames_x0"
    frames_zt_dir = out_dir / "frames_zt"
    frames_x0_dir.mkdir(parents=True, exist_ok=True)
    frames_zt_dir.mkdir(parents=True, exist_ok=True)

    device = pipe._execution_device
    dtype = pipe.unet_vton.dtype
    vae, scheduler = pipe.vae, pipe.scheduler

    pipe._image_guidance_scale = image_guidance_scale
    do_cfg = pipe.do_classifier_free_guidance
    generator = torch.Generator(device=device).manual_seed(seed)

    # 1. Garment CLIP embedding -> fake "text" embeds, exactly as inference_ootd_hd.py
    prompt_image = auto_processor(images=image_garm, return_tensors="pt").to(device)
    prompt_image = image_encoder(prompt_image.data["pixel_values"]).image_embeds.unsqueeze(1)
    prompt_embeds = pipe.text_encoder(pipe.tokenizer(
        [""], max_length=2, padding="max_length", truncation=True, return_tensors="pt",
    ).input_ids.to(device))[0]
    prompt_embeds[:, 1:] = prompt_image[:]
    prompt_embeds = pipe._encode_prompt(
        None, device, 1, do_cfg, None, prompt_embeds=prompt_embeds, negative_prompt_embeds=None,
    )

    # 2. Preprocess images/mask
    image_garm_t = pipe.image_processor.preprocess(image_garm)
    image_vton_t = pipe.image_processor.preprocess(image_vton)
    image_ori_t = pipe.image_processor.preprocess(image_ori)
    mask_arr = np.array(mask)
    mask_arr[mask_arr < 127] = 0
    mask_arr[mask_arr >= 127] = 255
    mask_t = torch.tensor(mask_arr) / 255
    mask_t = mask_t.reshape(-1, 1, mask_t.size(-2), mask_t.size(-1))

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    garm_latents = pipe.prepare_garm_latents(image_garm_t, 1, 1, prompt_embeds.dtype, device, do_cfg, generator)
    vton_latents, mask_latents, image_ori_latents = pipe.prepare_vton_latents(
        image_vton_t, mask_t, image_ori_t, 1, 1, prompt_embeds.dtype, device, do_cfg, generator,
    )
    height, width = vton_latents.shape[-2:]
    height, width = height * pipe.vae_scale_factor, width * pipe.vae_scale_factor

    latents = pipe.prepare_latents(1, vae.config.latent_channels, height, width, prompt_embeds.dtype, device, generator)
    noise = latents.clone()
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=1.0)

    # Garment reference features: computed ONCE on the clean garment latent
    # and reused as fixed keys/values every step (OOTDiffusion's "outfitting
    # fusion") -- unlike CatVTON, which re-derives garment context from the
    # (still-being-denoised) concatenated canvas at every step.
    _, spatial_attn_outputs = pipe.unet_garm(garm_latents, 0, encoder_hidden_states=prompt_embeds, return_dict=False)

    step_timesteps = []
    for i, t in enumerate(tqdm(timesteps, desc=out_dir.name)):
        z_t = latents  # noisy latent going INTO this step, pre-repaint-blend

        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        scaled_latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        latent_vton_model_input = torch.cat([scaled_latent_model_input, vton_latents], dim=1)

        noise_pred = pipe.unet_vton(
            latent_vton_model_input, spatial_attn_outputs.copy(), t,
            encoder_hidden_states=prompt_embeds, return_dict=False,
        )[0]

        if do_cfg:
            noise_pred_text_image, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_text + image_guidance_scale * (noise_pred_text_image - noise_pred_text)

        step_output = scheduler.step(noise_pred, t, latents, **extra_step_kwargs)

        if i == 0:
            x0_hat, _ = verify_pred_original_sample(scheduler, step_output, noise_pred, t, latents)
        else:
            x0_hat = getattr(step_output, "pred_original_sample", None)
            if x0_hat is None:
                alpha_bar_t = scheduler.alphas_cumprod.to(latents.device)[t]
                x0_hat = (latents - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()

        batch = torch.cat([z_t, x0_hat], dim=0)
        decoded = decode_latents(vae, batch, dtype)
        zt_pil, x0_pil = to_pil(decoded[0:1])[0], to_pil(decoded[1:2])[0]
        zt_pil.save(frames_zt_dir / f"step_{i:03d}.png")
        x0_pil.save(frames_x0_dir / f"step_{i:03d}.png")
        step_timesteps.append(int(t))

        latents = step_output.prev_sample

        # RePaint-style blend: keep the known (unmasked) region locked to the
        # original image renoised to the NEXT step's noise level, and let
        # only the masked (garment) region carry the model's own update.
        init_latents_proper = image_ori_latents * vae.config.scaling_factor
        if i < len(timesteps) - 1:
            init_latents_proper = scheduler.add_noise(init_latents_proper, noise, torch.tensor([timesteps[i + 1]]))
        latents = (1 - mask_latents) * init_latents_proper + mask_latents * latents

    final_img = decode_latents(vae, latents, dtype)
    to_pil(final_img)[0].save(out_dir / "final.png")

    return step_timesteps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--person", required=True)
    p.add_argument("--cloth", required=True)
    p.add_argument("--parse", default=None, help="defaults to <person's dataset>/image-parse-v3/<stem>.png")
    p.add_argument("--keypoints", default=None, help="defaults to <person's dataset>/openpose_json/<stem>_keypoints.json")
    p.add_argument("--run-name", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--image-guidance-scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--output-root", default=str(REPO_ROOT / "outputs"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    person_path = Path(args.person)
    dataset_root = person_path.parents[1]  # .../test/image/xxx.jpg -> .../test
    parse_path = Path(args.parse) if args.parse else dataset_root / "image-parse-v3" / f"{person_path.stem}.png"
    kp_path = Path(args.keypoints) if args.keypoints else dataset_root / "openpose_json" / f"{person_path.stem}_keypoints.json"

    out_dir = Path(args.output_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    person_image = Image.open(person_path).convert("RGB")
    cloth_image = Image.open(args.cloth).convert("RGB")

    mask, mask_gray = build_mask(person_image, parse_path, kp_path)
    mask.save(out_dir / "mask.png")
    masked_vton_img = Image.composite(mask_gray, person_image, mask)
    masked_vton_img.save(out_dir / "masked_vton.png")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    pipe = build_pipeline(dtype, args.device)
    auto_processor = AutoProcessor.from_pretrained(VIT_PATH)
    # torch_dtype matters here: left at the from_pretrained default (fp32)
    # this loads a second full-precision ~2.4GB copy of the vision tower
    # into CPU RAM on top of the already-resident pipeline, right before
    # moving it to the GPU -- a peak this machine's 7.6GB RAM can't always
    # absorb. Matching the rest of the pipeline's dtype halves that peak.
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH, torch_dtype=dtype).to(args.device)

    timesteps = run_with_capture(
        pipe, image_encoder, auto_processor,
        image_garm=cloth_image,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=person_image,
        num_inference_steps=args.steps,
        image_guidance_scale=args.image_guidance_scale,
        seed=args.seed,
        out_dir=out_dir,
    )

    config = {
        "model": "OOTDiffusion-HD",
        "unet_checkpoint": str(UNET_PATH),
        "scheduler": type(pipe.scheduler).__name__,
        "eta": 1.0,
        "weight_dtype": args.dtype,
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "guidance_scale": args.image_guidance_scale,
        "height": 1024,
        "width": 768,
        "category": CATEGORY,
        "inputs": {
            "person": str(person_path.resolve()),
            "cloth": str(Path(args.cloth).resolve()),
            "parse": str(parse_path.resolve()),
            "keypoints": str(kp_path.resolve()),
        },
        "timesteps": timesteps,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Done. Wrote {out_dir}")


if __name__ == "__main__":
    main()
