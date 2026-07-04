"""Train a real-time (feed-forward) style transfer network and stylise images.

Lab 4.2/4.3 equivalent. To keep it CPU-friendly and self-contained, the content
"dataset" is a set of procedurally-generated / cached images tiled into small
crops; the network learns to apply ONE style. After training, stylisation is a
single forward pass (real-time).

Usage:
    python scripts/train_realtime.py [--steps 200] [--size 256]
Outputs: results/realtime_stylized.png, results/realtime_meta.json,
         weights/transform_net.pth (gitignored)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aics.style_transfer.utils import (
    load_image,
    save_image,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from aics.style_transfer.realtime import train_transform_net, stylize


def make_content_pool(size, device, n=24, seed=0):
    """Build a pool of varied content tensors (normalised) for training.

    Uses the synthetic content generator with random crops/flips so the network
    sees diverse structure, not a single image.
    """
    from scripts.make_example_images import make_content

    base_path = "data/content.jpg"
    if not os.path.exists(base_path):
        make_content(max(size * 2, 512), base_path)
    big = load_image(base_path, size=int(size * 1.6), device="cpu")

    rng = np.random.default_rng(seed)
    _, _, H, W = big.shape
    pool = []
    for _ in range(n):
        top = rng.integers(0, max(1, H - size))
        left = rng.integers(0, max(1, W - size))
        crop = big[:, :, top : top + size, left : left + size]
        if crop.shape[2] != size or crop.shape[3] != size:
            crop = torch.nn.functional.interpolate(
                crop, size=(size, size), mode="bilinear", align_corners=False
            )
        if rng.random() < 0.5:
            crop = torch.flip(crop, dims=[3])
        pool.append(crop)
    return pool


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--style", default="data/style.jpg")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--style-weight", type=float, default=1e5)
    p.add_argument("--out", default="results/realtime_stylized.png")
    args = p.parse_args()

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    torch.manual_seed(0)
    os.makedirs("results", exist_ok=True)
    os.makedirs("weights", exist_ok=True)

    if not os.path.exists(args.style):
        from scripts.make_example_images import make_style
        make_style(args.size, args.style)

    device = "cpu"
    style = load_image(args.style, size=args.size, device=device)

    pool = make_content_pool(args.size, device, n=max(args.batch * 4, 16))

    def batches():
        # yield mini-batches, enough iterations to reach --steps
        rng = np.random.default_rng(123)
        count = 0
        while count < args.steps:
            idx = rng.choice(len(pool), size=args.batch, replace=False)
            yield torch.cat([pool[i] for i in idx], dim=0)
            count += 1

    print(f"Training TransformNet for real-time style transfer: "
          f"{args.steps} steps, batch {args.batch}, size {args.size}")
    t0 = time.time()
    model, history = train_transform_net(
        style,
        batches,
        epochs=1,
        style_weight=args.style_weight,
        lr=1e-3,
        device=device,
        log_every=20,
    )
    train_dt = time.time() - t0

    torch.save(model.state_dict(), "weights/transform_net.pth")

    # Real-time stylisation: time a single forward pass on a held-out content img.
    test_content = load_image("data/content.jpg", size=args.size, device=device)
    t1 = time.time()
    out = stylize(model, test_content, device=device)
    infer_dt = time.time() - t1
    save_image(out, args.out)
    save_image(test_content, "results/realtime_content.png")
    save_image(style, "results/realtime_style.png")

    meta = {
        "method": "Real-time feed-forward style transfer (Johnson et al.), VGG19 perceptual loss",
        "size": args.size,
        "train_steps": args.steps,
        "batch": args.batch,
        "style_weight": args.style_weight,
        "train_seconds": round(train_dt, 1),
        "inference_seconds_single_image": round(infer_dt, 3),
        "final_losses": history[-1] if history else None,
        "output": args.out,
        "weights": "weights/transform_net.pth",
    }
    with open("results/realtime_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nTrained in {train_dt:.1f}s. Single-image stylisation (forward pass) "
          f"= {infer_dt*1000:.1f} ms -> {args.out}")
    print("final losses:", meta["final_losses"])


if __name__ == "__main__":
    main()
