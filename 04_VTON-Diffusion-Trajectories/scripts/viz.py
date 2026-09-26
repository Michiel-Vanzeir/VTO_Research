"""
Build trajectory.mp4 and contact_sheet.png from a run's frames_x0/ directory.

Usage:
    python viz.py --run-dir outputs/pair1_solid
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_font(size=18):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def burn_overlay(img: Image.Image, step: int, total: int, timestep: int) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    text = f"step {step:03d}/{total-1:03d}  t={timestep}"
    font = load_font(max(14, img.width // 32))
    pad = 4
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    draw.rectangle([0, 0, tw + pad * 2, th + pad * 2], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 0), font=font)
    return img


def make_video(run_dir: Path, fps: int = 8):
    frames_dir = run_dir / "frames_x0"
    config = json.loads((run_dir / "run_config.json").read_text())
    timesteps = config["timesteps"]
    frame_paths = sorted(frames_dir.glob("step_*.png"))
    assert len(frame_paths) == len(timesteps), \
        f"{len(frame_paths)} frames but {len(timesteps)} logged timesteps"

    out_path = run_dir / "trajectory.mp4"
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8,
                                 macro_block_size=None)
    for i, (fp, t) in enumerate(zip(frame_paths, timesteps)):
        img = burn_overlay(Image.open(fp).convert("RGB"), i, len(frame_paths), t)
        writer.append_data(np.array(img))
    writer.close()
    print(f"Wrote {out_path} ({len(frame_paths)} frames @ {fps}fps)")


def make_contact_sheet(run_dir: Path, target_tiles: int = 14, cols: int = 4):
    frames_dir = run_dir / "frames_x0"
    config = json.loads((run_dir / "run_config.json").read_text())
    timesteps = config["timesteps"]
    frame_paths = sorted(frames_dir.glob("step_*.png"))
    n = len(frame_paths)

    step_n = max(1, round(n / target_tiles))
    indices = list(range(0, n, step_n))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    tiles = [burn_overlay(Image.open(frame_paths[i]).convert("RGB"), i, n, timesteps[i]) for i in indices]
    tw, th = tiles[0].size
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (tw * cols, th * rows), (20, 20, 20))
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        sheet.paste(tile, (c * tw, r * th))

    out_path = run_dir / "contact_sheet.png"
    sheet.save(out_path)
    print(f"Wrote {out_path} ({len(tiles)} tiles from {n} steps)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--tiles", type=int, default=14)
    p.add_argument("--cols", type=int, default=4)
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    make_video(run_dir, fps=args.fps)
    make_contact_sheet(run_dir, target_tiles=args.tiles, cols=args.cols)


if __name__ == "__main__":
    main()
