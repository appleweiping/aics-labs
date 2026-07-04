"""Module container: chains operators for forward/backward."""

from __future__ import annotations

import numpy as np


class Flatten:
    def __init__(self):
        self.shape = None

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def parameters(self):
        """Yield (layer, param_name, grad_name) for every trainable param."""
        for layer in self.layers:
            if hasattr(layer, "params"):
                for name, _, grad_name in layer.params():
                    yield layer, name, grad_name
