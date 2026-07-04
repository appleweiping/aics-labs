"""Generate example content + style images procedurally.

We avoid committing copyrighted photos/paintings. These synthetic images are
deterministic, license-free, and exercise the full style-transfer pipeline
(content has structure/edges; style has bold colour texture). Users can drop in
their own content.jpg / style.jpg to restyle real photos.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw


def make_content(size=256, path="data/content.jpg"):
    """A simple structured 'scene': horizon, sun, rolling hills, a house."""
    img = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(img)
    # sky gradient
    for y in range(size):
        t = y / size
        r = int(120 + 100 * (1 - t))
        g = int(160 + 60 * (1 - t))
        b = int(220 - 40 * t)
        d.line([(0, y), (size, y)], fill=(r, g, b))
    # sun
    d.ellipse([size * 0.65, size * 0.12, size * 0.85, size * 0.32], fill=(255, 240, 180))
    # hills
    hill_top = int(size * 0.6)
    d.rectangle([0, hill_top, size, size], fill=(70, 130, 70))
    xs = np.linspace(0, size, 60)
    ys = hill_top + 18 * np.sin(xs / size * 6)
    d.polygon(
        [(0, size), *[(x, y) for x, y in zip(xs, ys)], (size, size)],
        fill=(60, 110, 60),
    )
    # house
    hx, hy = int(size * 0.2), int(size * 0.66)
    hs = int(size * 0.14)
    d.rectangle([hx, hy, hx + hs, hy + hs], fill=(200, 180, 150))
    d.polygon([(hx - 8, hy), (hx + hs + 8, hy), (hx + hs // 2, hy - hs // 2)], fill=(150, 70, 60))
    d.rectangle([hx + hs // 3, hy + hs // 2, hx + hs // 2, hy + hs], fill=(90, 60, 40))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, quality=95)
    return path


def make_style(size=256, path="data/style.jpg"):
    """A bold, painterly style texture: swirling colour bands + brush blobs."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:size, 0:size]
    # swirling bands
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2)
    theta = np.arctan2(yy - size / 2, xx - size / 2)
    band = np.sin(r / 12 + theta * 3)
    red = (0.5 + 0.5 * np.sin(band * 3 + 0)) * 255
    grn = (0.5 + 0.5 * np.sin(band * 3 + 2)) * 255
    blu = (0.5 + 0.5 * np.sin(band * 3 + 4)) * 255
    arr = np.stack([red, grn, blu], axis=-1).astype(np.uint8)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    # brush blobs
    palette = [(230, 40, 40), (250, 200, 30), (30, 90, 200), (20, 160, 90), (240, 240, 240)]
    for _ in range(120):
        cx, cy = rng.integers(0, size, 2)
        rad = rng.integers(4, 16)
        col = palette[rng.integers(0, len(palette))]
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=col)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, quality=95)
    return path


if __name__ == "__main__":
    c = make_content()
    s = make_style()
    print("wrote", c, "and", s)
