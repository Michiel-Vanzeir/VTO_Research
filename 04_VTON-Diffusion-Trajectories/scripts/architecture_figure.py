"""
Small column-wide schematic of how the three try-on models inject the garment
into the denoising network (one row per model), for the related-work section.

Writes current_paper/figures/architectures.pdf (and a .png preview).

Usage:
    python scripts/architecture_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from compare_trajectories import MODELS

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "current_paper" / "figures"
COL_W = 3.35  # inches

plt.rcParams.update({"font.size": 6, "pdf.fonttype": 42})

INPUT = dict(fc="#f1f1f1", ec="#7a7a7a")
ENCODER = dict(fc="#ffffff", ec="#333333")
NOTE = dict(fc="none", ec="#7a7a7a", linestyle=(0, (3, 2)))


def box(ax, x, y, w, h, text, style, bold=False, lw=0.6, fontsize=6, color="black"):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                linewidth=lw, **style))
    ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=fontsize, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.0)


def arrow(ax, x0, y0, x1, y1, label=None, label_dy=0.1, label_dx=0.0, ha="center"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#333333", shrinkA=0, shrinkB=0, mutation_scale=6))
    if label:
        ax.text((x0 + x1) / 2 + label_dx, (y0 + y1) / 2 + label_dy, label, ha=ha, va="bottom",
                fontsize=5.3, style="italic", color="#333333")


def unet(ax, y, color, text):
    style = dict(fc=matplotlib.colors.to_rgba(color, 0.14), ec=color)
    box(ax, 7.0, y, 2.9, 0.62, text, style, bold=True, lw=0.9)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(COL_W, 2.25))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.55, 7.98)
    ax.axis("off")
    (_, cat_col), (_, ootd_col), (_, idm_col) = MODELS["catvton"], MODELS["ootd"], MODELS["idm"]
    enc_x, enc_w = 2.45, 2.55

    # CatVTON: noisy latent, mask and [masked person | garment] concatenated into one inpainting UNet.
    y = 7.15
    ax.text(0.0, y + 0.45, "CatVTON", fontsize=6.5, fontweight="bold", color=cat_col, va="bottom")
    box(ax, 0.0, y, 3.3, 0.62, "$z_t$, mask,\nmasked person $\\Vert$ garment", INPUT)
    arrow(ax, 3.3, y, 7.0, y, "channel concat", label_dy=0.05)
    ax.text(5.15, y - 0.1, "no text, no cross-attention", ha="center", va="top", fontsize=5.3, style="italic")
    unet(ax, y, cat_col, "Denoising UNet\nSD 1.5 inpainting")

    # OOTDiffusion: outfitting UNet (run once) into self-attention; CLIP image + category label
    # into cross-attention of both UNets; the mask only blends the latents after each step.
    y = 5.2
    ax.text(0.0, y + 1.22, "OOTDiffusion", fontsize=6.5, fontweight="bold", color=ootd_col, va="bottom")
    box(ax, 0.0, y + 0.4, 2.1, 0.45, "garment", INPUT)
    box(ax, 0.0, y - 0.85, 2.1, 0.62, "$z_t$, masked\nperson", INPUT)
    box(ax, enc_x, y + 0.85, enc_w, 0.55, "Outfitting UNet\nrun once, t = 0", ENCODER)
    box(ax, enc_x, y - 0.05, enc_w, 0.55, "CLIP image\n+ category label", ENCODER)
    arrow(ax, 2.1, y + 0.45, enc_x, y + 0.85)
    arrow(ax, 2.1, y + 0.35, enc_x, y - 0.05)
    arrow(ax, enc_x + 0.45, y + 0.225, enc_x + 0.45, y + 0.575, "cross-attention",
          label_dy=-0.08, label_dx=0.1, ha="left")
    arrow(ax, enc_x + enc_w, y + 0.85, 7.0, y + 0.2, "self-attention", label_dy=0.1)
    arrow(ax, enc_x + enc_w, y - 0.05, 7.0, y + 0.0, "cross-attention", label_dy=0.02)
    arrow(ax, 2.1, y - 0.85, 7.0, y - 0.2, "channel concat", label_dy=-0.4)
    unet(ax, y, ootd_col, "Denoising UNet\nSD 1.5")
    box(ax, 7.0, y - 0.78, 2.9, 0.55, "mask: latent blending\nafter each step", NOTE, fontsize=5.3, color="#333333")

    # IDM-VTON: GarmentNet (every step, own caption) into self-attention;
    # IP-Adapter and caption text into cross-attention.
    y = 2.1
    ax.text(0.0, y + 1.42, "IDM-VTON", fontsize=6.5, fontweight="bold", color=idm_col, va="bottom")
    box(ax, 0.0, y + 0.5, 2.1, 0.45, "garment", INPUT)
    box(ax, 0.0, y - 0.95, 2.1, 0.95, "$z_t$, mask,\nmasked person,\nDensePose", INPUT)
    box(ax, enc_x, y + 0.95, enc_w, 0.8, "GarmentNet\n+ caption text\nevery step t", ENCODER)
    box(ax, enc_x, y + 0.05, enc_w, 0.55, "IP-Adapter\n+ caption text", ENCODER)
    arrow(ax, 2.1, y + 0.55, enc_x, y + 0.95)
    arrow(ax, 2.1, y + 0.45, enc_x, y + 0.07)
    arrow(ax, enc_x + enc_w, y + 0.95, 7.0, y + 0.2, "self-attention", label_dy=0.16)
    arrow(ax, enc_x + enc_w, y + 0.05, 7.0, y + 0.0, "cross-attention", label_dy=0.02)
    arrow(ax, 2.1, y - 0.95, 7.0, y - 0.2, "channel concat", label_dy=-0.42)
    unet(ax, y, idm_col, "TryonNet\nSDXL inpainting")

    ax.text(9.9, 0.68, "$\\Vert$ = spatial concatenation", ha="right", va="center", fontsize=5.3, color="#333333")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUT / "architectures.pdf")
    fig.savefig(OUT / "architectures.png", dpi=300)
    plt.close(fig)
    print(f"Wrote {OUT / 'architectures.pdf'}")


if __name__ == "__main__":
    main()
