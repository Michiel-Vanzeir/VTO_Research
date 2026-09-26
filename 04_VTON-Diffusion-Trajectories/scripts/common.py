"""Helpers shared by every per-model trajectory-capture script.

Both CatVTON and OOTDiffusion are reimplemented step-by-step (see the module
docstring in each `*_trajectory.py`) rather than called as black-box
pipelines, so they can share the same latent->image decode, the same
manual x0-hat formula, and the same cross-check against whatever the
scheduler itself reports -- keeping the two models' trajectories directly
comparable.
"""
import torch
from PIL import Image


def decode_latents(vae, latents: torch.Tensor, weight_dtype) -> torch.Tensor:
    """Fixed, consistent latent->[0,1] image conversion. No per-frame
    normalization/contrast-stretch: that would destroy the exact
    high-noise-step low-contrast signal we're trying to observe."""
    latents = latents.to(vae.device, dtype=weight_dtype)
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


def to_pil(image_01: torch.Tensor):
    arr = image_01.cpu().permute(0, 2, 3, 1).float().numpy()
    arr = (arr * 255).round().astype("uint8")
    return [Image.fromarray(a) for a in arr]


def verify_pred_original_sample(scheduler, step_output, noise_pred, t, latents):
    """Cross-check scheduler's pred_original_sample against the manual
    x0_hat formula on one sample, per instructions not to blindly trust it."""
    alphas_cumprod = scheduler.alphas_cumprod.to(latents.device)
    alpha_bar_t = alphas_cumprod[t]
    manual_x0 = (latents - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
    pred = getattr(step_output, "pred_original_sample", None)
    if pred is None:
        print(f"[verify] step output has no pred_original_sample at t={int(t)}; using manual formula for this run.")
        return manual_x0, False
    diff = (manual_x0.float() - pred.float()).abs().max().item()
    ok = diff <= 1e-2
    print(f"[verify] t={int(t)} max|manual_x0 - pred_original_sample| = {diff:.6f} -> {'OK' if ok else 'MISMATCH, falling back to manual'}")
    return (pred if ok else manual_x0), ok


def x0_from_eps(scheduler, noise_pred: torch.Tensor, t, latents: torch.Tensor) -> torch.Tensor:
    """Manual x0-hat formula, for steps where we don't bother re-verifying
    against the scheduler (only step 0 is cross-checked per run)."""
    alpha_bar_t = scheduler.alphas_cumprod.to(latents.device)[t]
    return (latents - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
