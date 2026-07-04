"""2D convolution operator (forward + backward), im2col based."""

from __future__ import annotations

import numpy as np

from .im2col import im2col, col2im, get_conv_out_size


def conv2d_forward(x, weight, bias, stride=1, pad=0):
    """x: (N,C,H,W), weight: (F,C,kh,kw), bias: (F,).  Returns (out, cache)."""
    n, c, h, w = x.shape
    f, c_w, kh, kw = weight.shape
    assert c == c_w, "input/weight channel mismatch"

    cols, out_h, out_w = im2col(x, kh, kw, stride, pad)  # (N, C*kh*kw, L)
    w_col = weight.reshape(f, -1)  # (F, C*kh*kw)

    # (N, F, L)
    out = np.einsum("fk,nkl->nfl", w_col, cols) + bias.reshape(1, f, 1)
    out = out.reshape(n, f, out_h, out_w)

    cache = (x.shape, weight, bias, stride, pad, cols)
    return out, cache


def conv2d_backward(dout, cache):
    """Returns dx, dweight, dbias."""
    x_shape, weight, bias, stride, pad, cols = cache
    n, c, h, w = x_shape
    f, _, kh, kw = weight.shape
    _, _, out_h, out_w = dout.shape

    dout_r = dout.reshape(n, f, out_h * out_w)  # (N, F, L)

    # dbias
    dbias = dout_r.sum(axis=(0, 2))

    # dweight: sum over batch and spatial of dout * cols
    w_col = weight.reshape(f, -1)
    dw_col = np.einsum("nfl,nkl->fk", dout_r, cols)
    dweight = dw_col.reshape(weight.shape)

    # dx via col2im
    dcols = np.einsum("fk,nfl->nkl", w_col, dout_r)  # (N, C*kh*kw, L)
    dx = col2im(dcols, x_shape, kh, kw, stride, pad)
    return dx, dweight, dbias


class Conv2D:
    """Stateful conv layer with He initialisation."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, pad=0, seed=None):
        rng = np.random.default_rng(seed)
        self.stride = stride
        self.pad = pad
        fan_in = in_ch * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)
        self.weight = (rng.standard_normal((out_ch, in_ch, kernel_size, kernel_size)) * std)
        self.bias = np.zeros(out_ch)
        self.cache = None
        self.dweight = None
        self.dbias = None

    def forward(self, x):
        out, self.cache = conv2d_forward(x, self.weight, self.bias, self.stride, self.pad)
        return out

    def backward(self, dout):
        dx, self.dweight, self.dbias = conv2d_backward(dout, self.cache)
        return dx

    def params(self):
        return [("weight", self.weight, "dweight"), ("bias", self.bias, "dbias")]
