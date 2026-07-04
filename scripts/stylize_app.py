"""Capstone (Lab 7 equivalent): the full image style-transfer application.

A single CLI that stylises any content image with any style image, using either:
  --method gatys     : non-real-time, optimisation-based (higher quality, slow)
  --method realtime  : train a feed-forward net for the style, then one-pass stylise

This is the end-to-end deliverable the AICS labs build toward. It reuses the
operators/framework insight (conv, pooling, gram, backprop) via the PyTorch VGG19
extractor and produces a REAL stylised output image.

Examples:
    python scripts/stylize_app.py --content data/content.jpg --style data/style.jpg \
        --method gatys --steps 300 --out results/app_gatys.png
    python scripts/stylize_app.py --method realtime --steps 200 --out results/app_realtime.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aics.style_transfer.utils import load_image, save_image
from aics.style_transfer.gatys import run_style_transfer
from aics.style_transfer.realtime import train_transform_net, stylize
from aics.style_transfer.transform_net import TransformNet


def ensure_images(content_path, style_path, size):
    from scripts.make_example_images import make_content, make_style
    if not os.path.exists(content_path):
        make_content(size, content_path)
    if not os.path.exists(style_path):
        make_style(size, style_path)


def main():
    p = argparse.ArgumentParser(description="AICS style-transfer application")
    p.add_argument("--content", default="data/content.jpg")
    p.add_argument("--style", default="data/style.jpg")
    p.add_argument("--method", choices=["gatys", "realtime"], default="gatys")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--style-weight", type=float, default=None)
    p.add_argument("--out", default="results/app_output.png")
    p.add_argument("--weights", default="weights/transform_net.pth",
                   help="realtime: load pretrained transform net if present")
    args = p.parse_args()

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    torch.manual_seed(0)
    os.makedirs("results", exist_ok=True)
    ensure_images(args.content, args.style, args.size)

    device = "cpu"
    content = load_image(args.content, size=args.size, device=device)
    style = load_image(args.style, size=args.size, device=device)

    t0 = time.time()
    if args.method == "gatys":
        sw = args.style_weight if args.style_weight is not None else 1e6
        out, _ = run_style_transfer(
            content, style, num_steps=args.steps, style_weight=sw, device=device
        )
    else:  # realtime
        if os.path.exists(args.weights):
            print(f"loading pretrained transform net from {args.weights}")
            model = TransformNet()
            model.load_state_dict(
                torch.load(args.weights, map_location=device, weights_only=True)
            )
        else:
            print("no pretrained weights; training a transform net for this style...")
            from scripts.train_realtime import make_content_pool
            pool = make_content_pool(args.size, device, n=16)
            import numpy as np

            def batches():
                rng = np.random.default_rng(123)
                for _ in range(args.steps):
                    idx = rng.choice(len(pool), size=2, replace=False)
                    yield torch.cat([pool[i] for i in idx], dim=0)

            sw = args.style_weight if args.style_weight is not None else 1e5
            model, _ = train_transform_net(
                style, batches, epochs=1, style_weight=sw, device=device, log_every=50
            )
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), args.weights)
        out = stylize(model, content, device=device)

    dt = time.time() - t0
    save_image(out, args.out)
    print(f"\n[{args.method}] stylised {args.content} with {args.style} "
          f"in {dt:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
