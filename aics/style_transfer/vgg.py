"""VGG19 feature extractor for style transfer.

Uses torchvision's ImageNet-pretrained VGG19 (the same network the AICS style-
transfer lab uses as its fixed feature extractor). We expose the intermediate
conv activations used as content and style representations, and the Gram matrix
used to capture style statistics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

# Layer names -> indices in torchvision vgg19.features (conv-relu numbering).
# We use relu activations at these depths (Gatys et al. default choice).
STYLE_LAYERS = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]
CONTENT_LAYERS = ["relu4_2"]

# Map readable names to the sequential index of the *relu* output in vgg19.features.
_LAYER_INDEX = {
    "relu1_1": 1,
    "relu2_1": 6,
    "relu3_1": 11,
    "relu4_1": 20,
    "relu4_2": 22,
    "relu5_1": 29,
}


class VGGFeatures(nn.Module):
    def __init__(self, layers=None):
        super().__init__()
        if layers is None:
            layers = STYLE_LAYERS + CONTENT_LAYERS
        self.layers = layers
        self.wanted = {_LAYER_INDEX[name]: name for name in layers}
        try:
            weights = models.VGG19_Weights.IMAGENET1K_V1
            vgg = models.vgg19(weights=weights).features
        except Exception:
            vgg = models.vgg19(pretrained=True).features
        # Freeze; we only need forward features.
        self.features = vgg.eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        # Replace in-place ReLUs to be safe for autograd graph reuse.
        for m in self.features.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def forward(self, x):
        out = {}
        max_idx = max(self.wanted)
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.wanted:
                out[self.wanted[i]] = x
            if i >= max_idx:
                break
        return out


def gram_matrix(feature):
    """Gram matrix of a (N,C,H,W) feature map, normalised by element count."""
    n, c, h, w = feature.shape
    f = feature.view(n, c, h * w)
    g = torch.bmm(f, f.transpose(1, 2))
    return g / (c * h * w)
