"""im2col / col2im helpers.

Convolution is implemented as a matrix multiply after unrolling input patches
into columns (im2col). This is the standard trick used by real DL frameworks and
by the AICS convolution lab. Everything is vectorised numpy — no python loops
over spatial positions.
"""

from __future__ import annotations

import numpy as np


def get_conv_out_size(in_size: int, kernel: int, stride: int, pad: int) -> int:
    return (in_size + 2 * pad - kernel) // stride + 1


def im2col(x: np.ndarray, kh: int, kw: int, stride: int, pad: int) -> np.ndarray:
    """Turn a (N, C, H, W) tensor into a (N, C*kh*kw, out_h*out_w) column matrix.

    Uses stride-tricks free fancy indexing so it stays fully vectorised.
    """
    n, c, h, w = x.shape
    out_h = get_conv_out_size(h, kh, stride, pad)
    out_w = get_conv_out_size(w, kw, stride, pad)

    x_pad = np.pad(
        x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant"
    )

    # Build the index grids once.
    i0 = np.repeat(np.arange(kh), kw)
    i0 = np.tile(i0, c)
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(kw), kh * c)
    j1 = stride * np.tile(np.arange(out_w), out_h)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(c), kh * kw).reshape(-1, 1)

    cols = x_pad[:, k, i, j]  # (N, C*kh*kw, out_h*out_w)
    return cols, out_h, out_w


def col2im(
    cols: np.ndarray,
    x_shape: tuple[int, int, int, int],
    kh: int,
    kw: int,
    stride: int,
    pad: int,
) -> np.ndarray:
    """Inverse of im2col, accumulating overlapping gradients (for conv backward)."""
    n, c, h, w = x_shape
    out_h = get_conv_out_size(h, kh, stride, pad)
    out_w = get_conv_out_size(w, kw, stride, pad)

    h_pad, w_pad = h + 2 * pad, w + 2 * pad
    x_pad = np.zeros((n, c, h_pad, w_pad), dtype=cols.dtype)

    i0 = np.repeat(np.arange(kh), kw)
    i0 = np.tile(i0, c)
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(kw), kh * c)
    j1 = stride * np.tile(np.arange(out_w), out_h)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(c), kh * kw).reshape(-1, 1)

    np.add.at(x_pad, (slice(None), k, i, j), cols)

    if pad == 0:
        return x_pad
    return x_pad[:, :, pad:-pad, pad:-pad]
