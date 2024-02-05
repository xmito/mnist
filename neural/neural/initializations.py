import numpy as np


def default_weight_init(x, y):
    return np.random.randn(y, x) / np.sqrt(x)


def default_bias_init(y):
    return np.random.randn(y, 1)
