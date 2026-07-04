"""Lab 2/4 equivalent: train a LeNet-style CNN on MNIST using the FROM-SCRATCH
numpy framework (our own Conv2D/MaxPool2D/ReLU/Linear + SGD + backprop).

This proves the operators compose into a working, trainable network. CPU-only.
A subset is used to keep the pure-numpy conv fast enough on CPU; the network,
operators, and backprop are the real thing.

Run:
    python scripts/train_mnist_cnn.py
Outputs: results/mnist_cnn_history.json, results/mnist_cnn_curve.png
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aics.operators import Conv2D, MaxPool2D, ReLU, Linear
from aics.framework import Sequential, Flatten, SGD, train, evaluate
from scripts.download_mnist import load_mnist


def build_lenet():
    return Sequential(
        Conv2D(1, 6, kernel_size=5, stride=1, pad=2, seed=1),   # 28x28 -> 28x28
        ReLU(),
        MaxPool2D(2, stride=2),                                  # -> 14x14
        Conv2D(6, 16, kernel_size=5, stride=1, pad=0, seed=2),  # -> 10x10
        ReLU(),
        MaxPool2D(2, stride=2),                                  # -> 5x5
        Flatten(),
        Linear(16 * 5 * 5, 120, seed=3),
        ReLU(),
        Linear(120, 84, seed=4),
        ReLU(),
        Linear(84, 10, seed=5),
    )


def main():
    os.makedirs("results", exist_ok=True)
    x_tr, y_tr, x_te, y_te = load_mnist(flatten=False)

    # CPU-scale subset (pure-numpy conv). Still a real train/val/test split.
    n_train = int(os.environ.get("AICS_MNIST_TRAIN", "6000"))
    n_test = int(os.environ.get("AICS_MNIST_TEST", "2000"))
    rng = np.random.default_rng(0)
    tr_idx = rng.permutation(len(x_tr))[:n_train]
    te_idx = rng.permutation(len(x_te))[:n_test]
    x_tr, y_tr = x_tr[tr_idx], y_tr[tr_idx]
    x_te, y_te = x_te[te_idx], y_te[te_idx]

    # split a validation set out of train
    n_val = 1000
    x_val, y_val = x_tr[:n_val], y_tr[:n_val]
    x_trn, y_trn = x_tr[n_val:], y_tr[n_val:]

    model = build_lenet()
    opt = SGD(model, lr=0.05, momentum=0.9, weight_decay=1e-4)

    print(f"Training LeNet (from-scratch numpy ops) on {len(x_trn)} MNIST images...")
    history = train(
        model, opt, x_trn, y_trn, x_val, y_val,
        epochs=int(os.environ.get("AICS_MNIST_EPOCHS", "5")),
        batch_size=64, seed=0,
    )

    test_acc, test_loss = evaluate(model, x_te, y_te)
    print(f"\nFinal TEST accuracy = {test_acc:.4f}  (loss {test_loss:.4f})  on {len(x_te)} images")

    out = {
        "model": "LeNet (from-scratch numpy operators)",
        "n_train": len(x_trn),
        "n_val": len(x_val),
        "n_test": len(x_te),
        "history": history,
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
    }
    with open("results/mnist_cnn_history.json", "w") as f:
        json.dump(out, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history]
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, [h["train_loss"] for h in history], "o-", label="train loss")
        plt.plot(epochs, [h["val_loss"] for h in history], "s-", label="val loss")
        plt.plot(epochs, [h["val_acc"] for h in history], "^-", label="val acc")
        plt.xlabel("epoch")
        plt.title(f"LeNet on MNIST (from-scratch ops)\ntest acc = {test_acc:.3f}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/mnist_cnn_curve.png", dpi=110)
        print("saved results/mnist_cnn_curve.png")
    except Exception as e:  # pragma: no cover
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
