"""Run non-real-time (Gatys) style transfer and save a REAL stylised image.

Usage:
    python scripts/run_gatys.py [--content data/content.jpg] [--style data/style.jpg]
                                [--size 256] [--steps 300] [--out results/gatys_stylized.png]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aics.style_transfer.utils import load_image, save_image
from aics.style_transfer.gatys import run_style_transfer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--content", default="data/content.jpg")
    p.add_argument("--style", default="data/style.jpg")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--style-weight", type=float, default=1e6)
    p.add_argument("--out", default="results/gatys_stylized.png")
    args = p.parse_args()

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(args.content) or not os.path.exists(args.style):
        from scripts.make_example_images import make_content, make_style
        make_content(args.size, args.content)
        make_style(args.size, args.style)

    device = "cpu"
    content = load_image(args.content, size=args.size, device=device)
    style = load_image(args.style, size=args.size, device=device)

    print(f"content {tuple(content.shape)}  style {tuple(style.shape)}  steps={args.steps}")
    t0 = time.time()
    out, history = run_style_transfer(
        content, style,
        num_steps=args.steps,
        style_weight=args.style_weight,
        device=device,
    )
    dt = time.time() - t0
    save_image(out, args.out)
    # also save the content/style side for a comparison strip
    save_image(content, "results/gatys_content.png")
    save_image(style, "results/gatys_style.png")

    meta = {
        "method": "Gatys optimization-based (non-real-time) style transfer, VGG19",
        "content": args.content,
        "style": args.style,
        "size": args.size,
        "steps": args.steps,
        "style_weight": args.style_weight,
        "seconds": round(dt, 1),
        "final_losses": history[-1] if history else None,
        "output": args.out,
    }
    with open("results/gatys_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone in {dt:.1f}s -> {args.out}")
    print("final losses:", meta["final_losses"])


if __name__ == "__main__":
    main()
