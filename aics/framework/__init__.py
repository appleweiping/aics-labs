"""A small deep-learning framework built on the from-scratch operators.

Corresponds to AICS Lab 2 / Lab 4: assemble the operators into a mini framework
with a Sequential container, a training loop, and an SGD/Momentum optimizer, and
train a real network (MLP + LeNet-style CNN) on a real dataset (MNIST digits).
"""

from .module import Sequential, Flatten
from .optimizer import SGD
from .trainer import train, evaluate

__all__ = ["Sequential", "Flatten", "SGD", "train", "evaluate"]
