"""Lab 4.2/4.3 equivalent: real-time (feed-forward) style transfer.

train_transform_net trains a TransformNet to reproduce a single style using the
VGG19 perceptual loss (content + style Gram loss). stylize runs one forward pass
(the "real-time" prediction). CPU-only friendly config.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .transform_net import TransformNet
from .vgg import VGGFeatures, gram_matrix, STYLE_LAYERS, CONTENT_LAYERS
from .utils import IMAGENET_MEAN, IMAGENET_STD


def _normalize_batch(batch, device):
    return (batch - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


def train_transform_net(
    style_img,
    content_batches,
    epochs=1,
    style_weight=1e5,
    content_weight=1.0,
    tv_weight=1e-6,
    lr=1e-3,
    device="cpu",
    log_every=10,
    log_fn=print,
):
    """Train a TransformNet for one style.

    style_img: normalised (1,3,H,W) tensor.
    content_batches: iterable yielding normalised (B,3,H,W) content tensors.
        (Re-iterable: pass a list, or a callable returning a fresh iterator.)
    Returns (model, history).
    """
    model = TransformNet().to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    extractor = VGGFeatures().to(device)

    with torch.no_grad():
        style_feats = extractor(style_img.to(device))
        style_grams = {l: gram_matrix(style_feats[l]) for l in STYLE_LAYERS}

    def total_variation(x):
        return (
            (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
            + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        )

    history = []
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        batches = content_batches() if callable(content_batches) else content_batches
        for content in batches:
            content = content.to(device)
            optimizer.zero_grad()

            out = model(content)  # network output (un-normalised, ~[0,1] range-ish)
            out_norm = _normalize_batch(out, device)

            out_feats = extractor(out_norm)
            with torch.no_grad():
                content_feats = extractor(content)

            c_loss = content_weight * sum(
                F.mse_loss(out_feats[l], content_feats[l].detach()) for l in CONTENT_LAYERS
            )
            s_loss = 0.0
            for l in STYLE_LAYERS:
                g = gram_matrix(out_feats[l])
                s_loss = s_loss + F.mse_loss(g, style_grams[l].expand_as(g))
            s_loss = style_weight * s_loss
            tv = tv_weight * total_variation(out)
            loss = c_loss + s_loss + tv
            loss.backward()
            optimizer.step()

            step += 1
            if step % log_every == 0 or step == 1:
                rec = {
                    "step": step,
                    "total": float(loss.item()),
                    "content": float(c_loss.item()),
                    "style": float(s_loss.item()),
                    "seconds": round(time.time() - t0, 1),
                }
                history.append(rec)
                log_fn(
                    f"step {step:4d}  total={rec['total']:.3f}  "
                    f"content={rec['content']:.3f}  style={rec['style']:.3f}"
                )
    return model, history


@torch.no_grad()
def stylize(model, content, device="cpu"):
    """One forward pass = real-time stylisation. content: normalised (1,3,H,W).

    Returns a normalised tensor ready for utils.save_image.
    """
    model.eval()
    out = model(content.to(device))
    # out is roughly in [0,1]; renormalise into the ImageNet space save_image expects
    out = (out - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return out
