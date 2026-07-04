"""Training / evaluation loop for the mini framework."""

from __future__ import annotations

import time

import numpy as np

from ..operators.losses import softmax, cross_entropy, softmax_cross_entropy_backward


def evaluate(model, x, y, batch_size=256):
    n = x.shape[0]
    correct = 0
    loss_sum = 0.0
    for i in range(0, n, batch_size):
        xb = x[i : i + batch_size]
        yb = y[i : i + batch_size]
        logits = model.forward(xb)
        probs = softmax(logits)
        loss_sum += cross_entropy(probs, yb) * xb.shape[0]
        correct += (probs.argmax(axis=1) == yb).sum()
    return correct / n, loss_sum / n


def train(
    model,
    optimizer,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs=3,
    batch_size=64,
    seed=0,
    log_fn=print,
):
    rng = np.random.default_rng(seed)
    n = x_train.shape[0]
    history = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        x_train, y_train = x_train[perm], y_train[perm]
        t0 = time.time()
        running = 0.0
        for i in range(0, n, batch_size):
            xb = x_train[i : i + batch_size]
            yb = y_train[i : i + batch_size]
            logits = model.forward(xb)
            probs = softmax(logits)
            running += cross_entropy(probs, yb) * xb.shape[0]
            dout = softmax_cross_entropy_backward(probs, yb)
            model.backward(dout)
            optimizer.step()
        train_loss = running / n
        val_acc, val_loss = evaluate(model, x_val, y_val)
        dt = time.time() - t0
        rec = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "seconds": round(dt, 1),
        }
        history.append(rec)
        log_fn(
            f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  ({dt:.1f}s)"
        )
    return history
