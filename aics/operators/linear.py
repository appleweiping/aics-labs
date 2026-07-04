"""Fully-connected (dense) operator (forward + backward)."""

from __future__ import annotations

import numpy as np


class Linear:
    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.default_rng(seed)
        std = np.sqrt(2.0 / in_features)
        self.weight = rng.standard_normal((in_features, out_features)) * std
        self.bias = np.zeros(out_features)
        self.cache = None
        self.dweight = None
        self.dbias = None

    def forward(self, x):
        self.cache = x
        return x @ self.weight + self.bias

    def backward(self, dout):
        x = self.cache
        self.dweight = x.T @ dout
        self.dbias = dout.sum(axis=0)
        dx = dout @ self.weight.T
        return dx

    def params(self):
        return [("weight", self.weight, "dweight"), ("bias", self.bias, "dbias")]
