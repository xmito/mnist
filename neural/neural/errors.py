import numpy as np


class SquaredError:
    def __call__(self, y_pred, y_true):
        return 0.5 * (np.linalg.norm(y_true - y_pred)**2 / y_true.shape[1])

    def d(self, y_pred, y_true):
        return y_pred - y_true

squared_error = SquaredError()


class CrossEntropy:
    def __call__(self, y_pred, y_true):
        return np.average(
            np.nan_to_num(
                -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred),
                keepdims=True,
                axis=1
            )
        )

    def d(self, y_pred, y_true):
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

cross_entropy = CrossEntropy()