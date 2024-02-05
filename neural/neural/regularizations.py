import numpy as np


class DefaultRegularization:
    def __call__(self, *args, **kwargs):
        return 0

    def d(self, *args, **kwargs):
        return 0

default_regularization = DefaultRegularization()


class L2Regularization:
    def __init__(self, lmbda):
        self.lmbda = lmbda

    def __call__(self, w, n):
        return (self.lmbda / (2 * n)) * np.linalg.norm(w)**2

    def d(self, w, n):
        return (self.lmbda / n) * w


class L1Regularization:
    def __init__(self, lmbda):
        self.lmbda = lmbda

    def __call__(self, w, n):
        return (self.lmbda / n) * np.sum(np.abs(w))

    def d(self, w, n):
        return (self.lmbda / n) * np.sign(w)
