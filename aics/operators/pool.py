"""Max / average pooling operators (forward + backward)."""

from __future__ import annotations

import numpy as np

from .im2col import get_conv_out_size


class MaxPool2D:
    def __init__(self, kernel_size, stride=None):
        self.k = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.cache = None

    def forward(self, x):
        n, c, h, w = x.shape
        k, s = self.k, self.stride
        out_h = get_conv_out_size(h, k, s, 0)
        out_w = get_conv_out_size(w, k, s, 0)

        # Reshape into windows via as_strided-free indexing.
        out = np.empty((n, c, out_h, out_w), dtype=x.dtype)
        argmax = np.empty((n, c, out_h, out_w, 2), dtype=np.int64)
        for i in range(out_h):
            for j in range(out_w):
                window = x[:, :, i * s : i * s + k, j * s : j * s + k]
                flat = window.reshape(n, c, -1)
                idx = flat.argmax(axis=2)
                out[:, :, i, j] = np.take_along_axis(flat, idx[..., None], axis=2)[..., 0]
                argmax[:, :, i, j, 0] = idx // k
                argmax[:, :, i, j, 1] = idx % k
        self.cache = (x.shape, argmax)
        return out

    def backward(self, dout):
        x_shape, argmax = self.cache
        n, c, h, w = x_shape
        k, s = self.k, self.stride
        _, _, out_h, out_w = dout.shape
        dx = np.zeros(x_shape, dtype=dout.dtype)
        nn, cc = np.meshgrid(np.arange(n), np.arange(c), indexing="ij")
        for i in range(out_h):
            for j in range(out_w):
                di = argmax[:, :, i, j, 0]
                dj = argmax[:, :, i, j, 1]
                dx[nn, cc, i * s + di, j * s + dj] += dout[:, :, i, j]
        return dx


class AvgPool2D:
    def __init__(self, kernel_size, stride=None):
        self.k = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.cache = None

    def forward(self, x):
        n, c, h, w = x.shape
        k, s = self.k, self.stride
        out_h = get_conv_out_size(h, k, s, 0)
        out_w = get_conv_out_size(w, k, s, 0)
        out = np.empty((n, c, out_h, out_w), dtype=x.dtype)
        for i in range(out_h):
            for j in range(out_w):
                window = x[:, :, i * s : i * s + k, j * s : j * s + k]
                out[:, :, i, j] = window.mean(axis=(2, 3))
        self.cache = (x.shape,)
        return out

    def backward(self, dout):
        (x_shape,) = self.cache
        n, c, h, w = x_shape
        k, s = self.k, self.stride
        _, _, out_h, out_w = dout.shape
        dx = np.zeros(x_shape, dtype=dout.dtype)
        for i in range(out_h):
            for j in range(out_w):
                dx[:, :, i * s : i * s + k, j * s : j * s + k] += (
                    dout[:, :, i, j][:, :, None, None] / (k * k)
                )
        return dx
