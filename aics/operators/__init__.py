"""From-scratch neural-network operators (numpy) with forward + backward.

These correspond to AICS Lab 2 / Lab 3.1: implement the core operators of a deep
learning framework (fully-connected, convolution, pooling, activations, softmax +
cross-entropy) purely in numpy, verified by finite-difference gradient checks.

The original course targets the DLP/Cambricon simulator (pycnml / BCL). That
toolchain is unavailable on this machine, so the *equivalent* operators are
implemented here in numpy on CPU (see README, "Deviations").
"""

from .conv import Conv2D, conv2d_forward, conv2d_backward
from .pool import MaxPool2D, AvgPool2D
from .activation import ReLU, Sigmoid, Tanh
from .linear import Linear
from .losses import softmax, cross_entropy, softmax_cross_entropy_backward

__all__ = [
    "Conv2D",
    "conv2d_forward",
    "conv2d_backward",
    "MaxPool2D",
    "AvgPool2D",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Linear",
    "softmax",
    "cross_entropy",
    "softmax_cross_entropy_backward",
]
