"""Lab 3.3 equivalent: non-real-time (optimization-based) neural style transfer.

Gatys et al. 2016 "Image Style Transfer Using Convolutional Neural Networks".
The output image pixels are directly optimised (L-BFGS) to match the content
image's deep features and the style image's Gram-matrix statistics, using the
frozen VGG19 extractor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .vgg import VGGFeatures, gram_matrix, STYLE_LAYERS, CONTENT_LAYERS


def run_style_transfer(
    content,
    style,
    num_steps=300,
    style_weight=1e6,
    content_weight=1.0,
    tv_weight=1e-4,
    device="cpu",
    log_every=50,
    log_fn=print,
):
    """Return the stylised image tensor and a list of loss records."""
    extractor = VGGFeatures().to(device)

    with torch.no_grad():
        content_feats = extractor(content)
        style_feats = extractor(style)
        style_grams = {l: gram_matrix(style_feats[l]) for l in STYLE_LAYERS}
        content_targets = {l: content_feats[l].detach() for l in CONTENT_LAYERS}

    # Initialise from the content image. .contiguous() so LBFGS can flatten grads.
    img = content.clone().contiguous().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [img], max_iter=num_steps, lr=1.0, line_search_fn="strong_wolfe"
    )

    history = []
    step = {"n": 0}

    def total_variation(x):
        return (
            (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
            + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        )

    def closure():
        optimizer.zero_grad()
        feats = extractor(img)
        c_loss = sum(F.mse_loss(feats[l], content_targets[l]) for l in CONTENT_LAYERS)
        s_loss = sum(
            F.mse_loss(gram_matrix(feats[l]), style_grams[l]) for l in STYLE_LAYERS
        )
        tv = total_variation(img)
        loss = content_weight * c_loss + style_weight * s_loss + tv_weight * tv
        loss.backward()
        step["n"] += 1
        if step["n"] % log_every == 0 or step["n"] == 1:
            rec = {
                "step": step["n"],
                "total": float(loss.item()),
                "content": float((content_weight * c_loss).item()),
                "style": float((style_weight * s_loss).item()),
                "tv": float((tv_weight * tv).item()),
            }
            history.append(rec)
            log_fn(
                f"step {step['n']:4d}  total={rec['total']:.2f}  "
                f"content={rec['content']:.3f}  style={rec['style']:.2f}"
            )
        return loss

    # LBFGS runs num_steps internal iterations from a single .step() call.
    optimizer.step(closure)

    return img.detach(), history
