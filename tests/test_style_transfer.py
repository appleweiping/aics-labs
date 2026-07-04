"""Tests for the style-transfer building blocks.

Covers the pieces that do NOT need the (large) pretrained VGG download:
- gram_matrix correctness + symmetry
- TransformNet preserves spatial size and is differentiable
- image round-trip (load/save normalisation is invertible)
"""

import numpy as np
import torch

from aics.style_transfer.vgg import gram_matrix
from aics.style_transfer.transform_net import TransformNet
from aics.style_transfer.utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    tensor_to_image,
)


def test_gram_matrix_symmetric_and_correct():
    torch.manual_seed(0)
    f = torch.randn(2, 4, 5, 5)
    g = gram_matrix(f)
    assert g.shape == (2, 4, 4)
    # symmetric
    assert torch.allclose(g, g.transpose(1, 2), atol=1e-6)
    # matches manual definition for sample 0
    n, c, h, w = f.shape
    flat = f[0].view(c, -1)
    manual = (flat @ flat.t()) / (c * h * w)
    assert torch.allclose(g[0], manual, atol=1e-6)


def test_transform_net_shape_preserved():
    net = TransformNet().eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (1, 3, 128, 128)


def test_transform_net_differentiable():
    net = TransformNet()
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    y = net(x)
    loss = y.mean()
    loss.backward()
    # gradient flows to input and to first conv weights
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
    w = net.conv1.conv.weight
    assert w.grad is not None
    assert w.grad.abs().sum() > 0


def test_image_normalisation_roundtrip():
    # a normalised tensor in [0,1] image space should recover its pixels
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(1, 3, 8, 8)).astype(np.float32) / 255.0
    t = torch.from_numpy(pixels)
    norm = (t - IMAGENET_MEAN) / IMAGENET_STD
    recovered = tensor_to_image(norm)  # uint8 HWC
    back = recovered.astype(np.float32) / 255.0
    orig_hwc = pixels[0].transpose(1, 2, 0)
    assert np.abs(back - orig_hwc).max() < 0.01
