"""Softmax + cross-entropy loss (numerically stable)."""

from __future__ import annotations

import numpy as np


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, labels):
    """probs: (N, C) softmax outputs, labels: (N,) int class indices."""
    n = probs.shape[0]
    eps = 1e-12
    return -np.log(probs[np.arange(n), labels] + eps).mean()


def softmax_cross_entropy_backward(probs, labels):
    """Gradient of mean cross-entropy w.r.t. logits: (softmax - onehot) / N."""
    n = probs.shape[0]
    grad = probs.copy()
    grad[np.arange(n), labels] -= 1.0
    return grad / n
