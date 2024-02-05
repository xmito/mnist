import numpy as np


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def d(self, x):
        return self(x) * (1 - self(x))

sigmoid = Sigmoid()
