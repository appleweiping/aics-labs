"""Image I/O and pre/post-processing for style transfer."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

# ImageNet normalisation used by VGG19.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_image(path, size=None, device="cpu"):
    """Load an image as a normalised (1,3,H,W) tensor."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        if isinstance(size, int):
            w, h = img.size
            scale = size / max(w, h)
            new_size = (round(w * scale), round(h * scale))
            img = img.resize(new_size, Image.LANCZOS)
        else:
            img = img.resize(size, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t.to(device)


def tensor_to_image(t):
    """Inverse of load_image: normalised tensor -> uint8 HWC array."""
    t = t.detach().cpu()
    t = t * IMAGENET_STD + IMAGENET_MEAN
    t = t.clamp(0, 1)
    arr = (t.squeeze(0).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return arr


def save_image(t, path):
    arr = tensor_to_image(t)
    Image.fromarray(arr).save(path)
    return path
