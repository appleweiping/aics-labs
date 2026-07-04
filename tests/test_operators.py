"""Finite-difference gradient checks for every from-scratch operator.

This is the primary correctness verification for the AICS operator / framework
labs: for each operator we compare the analytic backward pass against a numerical
gradient computed by central differences. If backward is wrong, these fail.
"""

import numpy as np
import pytest

from aics.operators import (
    Conv2D,
    MaxPool2D,
    AvgPool2D,
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
    softmax,
    cross_entropy,
    softmax_cross_entropy_backward,
)


def numeric_grad(f, x, dout, eps=1e-5):
    """Central-difference gradient of sum(f(x) * dout) w.r.t. x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = (f(x) * dout).sum()
        x[idx] = old - eps
        fm = (f(x) * dout).sum()
        x[idx] = old
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def rel_error(a, b):
    return np.max(np.abs(a - b) / (np.maximum(1e-8, np.abs(a) + np.abs(b))))


def test_linear_grad():
    rng = np.random.default_rng(0)
    layer = Linear(5, 3, seed=1)
    x = rng.standard_normal((4, 5))
    out = layer.forward(x)
    dout = rng.standard_normal(out.shape)
    dx = layer.backward(dout)

    dx_num = numeric_grad(lambda z: layer.forward(z), x.copy(), dout)
    assert rel_error(dx, dx_num) < 1e-6

    # weight grad
    def f_w(w):
        layer.weight = w
        return layer.forward(x)

    dw_num = numeric_grad(f_w, layer.weight.copy(), dout)
    layer.forward(x)
    layer.backward(dout)
    assert rel_error(layer.dweight, dw_num) < 1e-6


def test_relu_grad():
    rng = np.random.default_rng(1)
    layer = ReLU()
    x = rng.standard_normal((3, 4)) * 2
    out = layer.forward(x)
    dout = rng.standard_normal(out.shape)
    dx = layer.backward(dout)
    dx_num = numeric_grad(lambda z: layer.forward(z), x.copy(), dout)
    assert rel_error(dx, dx_num) < 1e-6


def test_sigmoid_grad():
    rng = np.random.default_rng(2)
    layer = Sigmoid()
    x = rng.standard_normal((3, 4))
    out = layer.forward(x)
    dout = rng.standard_normal(out.shape)
    dx = layer.backward(dout)
    dx_num = numeric_grad(lambda z: layer.forward(z), x.copy(), dout)
    assert rel_error(dx, dx_num) < 1e-6


def test_tanh_grad():
    rng = np.random.default_rng(3)
    layer = Tanh()
    x = rng.standard_normal((3, 4))
    out = layer.forward(x)
    dout = rng.standard_normal(out.shape)
    dx = layer.backward(dout)
    dx_num = numeric_grad(lambda z: layer.forward(z), x.copy(), dout)
    assert rel_error(dx, dx_num) < 1e-6


def test_conv_grad_input():
    rng = np.random.default_rng(4)
    layer = Conv2D(2, 3, kernel_size=3, stride=1, pad=1, seed=5)
    x = rng.standard_normal((2, 2, 5, 5))
    out = layer.forward(x)
    dout = rng.standard_normal(out.shape)
    dx = layer.backward(dout)
    dx_num = numeric_grad(lambda z: layer.forward(z), x.copy(), dout)
    assert rel_error(dx, dx_num) < 1e-5


def test_conv_grad_weight_bias():
    rng = np.random.default_rng(6)
    layer = Conv2D(2, 3, kernel_size=3, stride=2, pad=1, seed=7)
    x = rng.standard_normal((2, 2, 6, 6))
    out = layer.forward(x)
    dout = rng.standard_normal(out.shape)
    layer.backward(dout)

    def f_w(w):
        layer.weight = w
        return layer.forward(x)

    dw_num = numeric_grad(f_w, layer.weight.copy(), dout)
    layer.forward(x)
    layer.backward(dout)
    assert rel_error(layer.dweight, dw_num) < 1e-5

    def f_b(b):
        layer.bias = b
        return layer.forward(x)

    db_num = numeric_grad(f_b, layer.bias.copy(), dout)
    layer.forward(x)
    layer.backward(dout)
    assert rel_error(layer.dbias, db_num) < 1e-5


def test_maxpool_grad():
    rng = np.random.default_rng(8)
    layer = MaxPool2D(2, stride=2)
    x = rng.standard_normal((2, 3, 4, 4))
    out = layer.forward(x)
    dout = rng.standard_normal(out.shape)
    dx = layer.backward(dout)
    dx_num = numeric_grad(lambda z: layer.forward(z), x.copy(), dout)
    assert rel_error(dx, dx_num) < 1e-5


def test_avgpool_grad():
    rng = np.random.default_rng(9)
    layer = AvgPool2D(2, stride=2)
    x = rng.standard_normal((2, 3, 4, 4))
    out = layer.forward(x)
    dout = rng.standard_normal(out.shape)
    dx = layer.backward(dout)
    dx_num = numeric_grad(lambda z: layer.forward(z), x.copy(), dout)
    assert rel_error(dx, dx_num) < 1e-5


def test_softmax_cross_entropy_grad():
    rng = np.random.default_rng(10)
    logits = rng.standard_normal((5, 4))
    labels = rng.integers(0, 4, size=5)
    probs = softmax(logits)
    grad = softmax_cross_entropy_backward(probs, labels)

    def loss(z):
        return cross_entropy(softmax(z), labels)

    num = np.zeros_like(logits)
    eps = 1e-5
    it = np.nditer(logits, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = logits[idx]
        logits[idx] = old + eps
        fp = loss(logits)
        logits[idx] = old - eps
        fm = loss(logits)
        logits[idx] = old
        num[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    assert rel_error(grad, num) < 1e-5


def test_conv_matches_reference():
    """Cross-check conv2d forward against a naive loop reference."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal((1, 2, 5, 5))
    layer = Conv2D(2, 2, 3, stride=1, pad=0, seed=12)
    out = layer.forward(x)

    # naive reference
    f, c, kh, kw = layer.weight.shape
    ref = np.zeros_like(out)
    for oc in range(f):
        for i in range(out.shape[2]):
            for j in range(out.shape[3]):
                patch = x[0, :, i : i + kh, j : j + kw]
                ref[0, oc, i, j] = (patch * layer.weight[oc]).sum() + layer.bias[oc]
    assert np.allclose(out, ref, atol=1e-10)
