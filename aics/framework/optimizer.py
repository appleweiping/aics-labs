"""SGD with optional momentum and weight decay."""

from __future__ import annotations

import numpy as np


class SGD:
    def __init__(self, model, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = {}

    def step(self):
        for layer, name, grad_name in self.model.parameters():
            param = getattr(layer, name)
            grad = getattr(layer, grad_name)
            if grad is None:
                continue
            if self.weight_decay:
                grad = grad + self.weight_decay * param
            if self.momentum:
                key = id(param)
                v = self._velocity.get(key, np.zeros_like(param))
                v = self.momentum * v - self.lr * grad
                self._velocity[key] = v
                param += v
            else:
                param -= self.lr * grad
