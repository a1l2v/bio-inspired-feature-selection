"""Prepare the wave images the way the reference cover (Untitled presentation.svg) uses them.

The reference places the paper-wave photo at 50 % opacity (alpha 127) inside a rectangle whose top-left and
bottom-right corners are rounded, so the grey slide background shows through and the image blends in. This
script bakes the crop, the corner mask and the opacity into PNGs that the deck builder places directly.

    cover_wave.png    crop of wave_vertical.jpg exactly as in the SVG (x 484-1155, y 447-2048 of 2048 px),
                      corner radius 111.8 px; placed at 1.96" x 4.68"
    problem_wave.png  centre crop of wave_horizontal.jpg to the 3.45" x 2.6" box, same corner radius in inches

Usage (from project root):
    .venv\\Scripts\\python.exe presentation\\prepare_images.py
"""

import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
ALPHA = 127  # 50 % opacity, as in the reference


def diagonal_round(img, radius):
    """Return img as RGBA with ALPHA opacity inside a rectangle whose top-left and bottom-right corners are rounded."""
    w, h = img.size
    scale = 4  # supersample the mask for smooth corners
    mask = Image.new("L", (w * scale, h * scale), 0)
    d = ImageDraw.Draw(mask)
    r = int(radius * scale)
    d.rounded_rectangle([0, 0, w * scale - 1, h * scale - 1], radius=r, fill=ALPHA,
                        corners=(True, False, True, False))  # top-left, top-right, bottom-right, bottom-left
    out = img.convert("RGBA")
    out.putalpha(mask.resize((w, h), Image.LANCZOS))
    return out


def main():
    src = Image.open(os.path.join(ASSETS, "wave_vertical.jpg"))
    cover = src.crop((484, 447, 1155, 2048))
    diagonal_round(cover, 111.8).save(os.path.join(ASSETS, "cover_wave.png"), optimize=True)

    src = Image.open(os.path.join(ASSETS, "wave_horizontal.jpg"))
    box_w, box_h = 3.45, 2.6
    crop_w = round(src.height * box_w / box_h)
    left = (src.width - crop_w) // 2
    problem = src.crop((left, 0, left + crop_w, src.height))
    px_per_in = crop_w / box_w
    diagonal_round(problem, 0.29 * px_per_in).save(os.path.join(ASSETS, "problem_wave.png"), optimize=True)
    print("saved cover_wave.png", cover.size, "and problem_wave.png", problem.size)


if __name__ == "__main__":
    main()
