"""Download MNIST via torchvision and cache it as .npz (gitignored).

MNIST is the real dataset used by the AICS Lab 2 framework exercise. It is
fetched at runtime and not committed (per factory spec).
"""

from __future__ import annotations

import os

import numpy as np


def load_mnist(data_dir="data", flatten=False):
    """Return (x_train, y_train, x_test, y_test) as numpy arrays in [0,1]."""
    os.makedirs(data_dir, exist_ok=True)
    cache = os.path.join(data_dir, "mnist.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        x_tr, y_tr, x_te, y_te = d["x_tr"], d["y_tr"], d["x_te"], d["y_te"]
    else:
        from torchvision import datasets  # lazy import

        tr = datasets.MNIST(data_dir, train=True, download=True)
        te = datasets.MNIST(data_dir, train=False, download=True)
        x_tr = tr.data.numpy().astype(np.float32) / 255.0
        y_tr = tr.targets.numpy().astype(np.int64)
        x_te = te.data.numpy().astype(np.float32) / 255.0
        y_te = te.targets.numpy().astype(np.int64)
        np.savez_compressed(cache, x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te)

    if flatten:
        x_tr = x_tr.reshape(x_tr.shape[0], -1)
        x_te = x_te.reshape(x_te.shape[0], -1)
    else:
        x_tr = x_tr[:, None, :, :]  # NCHW
        x_te = x_te[:, None, :, :]
    return x_tr, y_tr, x_te, y_te


if __name__ == "__main__":
    x_tr, y_tr, x_te, y_te = load_mnist()
    print("train", x_tr.shape, y_tr.shape, "test", x_te.shape, y_te.shape)
