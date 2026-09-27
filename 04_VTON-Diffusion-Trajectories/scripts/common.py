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


# --------------------------------------------------------------------- forks
# Fork experiment (fork_analysis.py): run the same pair many times, identical
# (seed `seed`) up to step k, then continue with a different noise source per
# fork seed. With DDIM eta=1 the scheduler injects fresh noise every step, so
# that is the only thing that differs after k. Whatever all forks from step k
# still agree on was already DECIDED at step k. k = 0 forks also draw
# different initial noise: the "nothing decided yet" baseline.

def fork_generators(device, seed: int, fork_step, fork_seed):
    """(generator for initial noise + steps before k, generator for steps >= k or None).
    In fork mode this also re-seeds the GLOBAL torch RNG with the base seed:
    CatVTON and IDM-VTON VAE-encode their conditioning images with
    latent_dist.sample(), which draws from it, so without this every fork
    (and every k) would be conditioned on slightly different latents."""
    if fork_step is not None:
        torch.manual_seed(seed)
    main = torch.Generator(device=device).manual_seed(fork_seed if fork_step == 0 else seed)
    fork = torch.Generator(device=device).manual_seed(fork_seed) if fork_step else None
    return main, fork


def step_kwargs(i: int, extra_step_kwargs: dict, fork_step, fork_gen) -> dict:
    """scheduler.step kwargs for step i: the fork's own noise from step k on."""
    if fork_gen is None or i < fork_step:
        return extra_step_kwargs
    return {**extra_step_kwargs, "generator": fork_gen}


def add_fork_args(parser):
    parser.add_argument("--fork-steps", default=None,
                        help="comma-separated fork steps, e.g. 0,5,10,20; enables fork mode (no per-step frames)")
    parser.add_argument("--fork-seeds", default="1,2,3,4", help="comma-separated noise seeds per fork step")


def run_forks(args, out_dir, run_one, config: dict):
    """Fork mode driver: run_one(fork_step, fork_seed) -> final PIL image.
    Resumable: forks whose image already exists are skipped. fork_config.json
    is written last and marks the pair as complete."""
    import json
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [int(x) for x in args.fork_steps.split(",")]
    seeds = [int(x) for x in args.fork_seeds.split(",")]
    for k in steps:
        for s in seeds:
            path = out_dir / f"k{k:02d}_s{s}.png"
            if path.exists():
                continue
            run_one(k, s).save(path)
            print(f"[fork] {out_dir.name}: k={k} seed={s} -> {path.name}", flush=True)
    (out_dir / "fork_config.json").write_text(json.dumps(
        {**config, "fork_steps": steps, "fork_seeds": seeds, "base_seed": args.seed}, indent=2))
    print(f"Done. Wrote {out_dir}")
