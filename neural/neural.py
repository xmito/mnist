import sys
import math
import logging

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def d(self, x):
        return self(x) * (1 - self(x))

sigmoid = Sigmoid()

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

def d(*args, fun=sigmoid):
    return fun.d(*args)

def default_weight_init(x, y):
    return np.random.randn(y, x) / np.sqrt(x)

def default_bias_init(y):
    return np.random.randn(y, 1)

class Neural:
    def __init__(
        self,
        sizes,
        cost=squared_error,
        weight_init=default_weight_init,
        bias_init=default_bias_init,
        regularization=default_regularization,
    ):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.regularization = regularization
        self.weights = [weight_init(x, y) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [bias_init(y) for y in sizes[1:]]

    @property
    def _hidden_layers(self):
        return self.num_layers - 2

    def _feed_forward(self, x, act=sigmoid):
        for w, b in zip(self.weights, self.biases):
            w_sum = np.dot(w, x) + b
            x = act(w_sum)
            yield w_sum, x

    def feed_forward(self, x, act=sigmoid):
        for _, x in self._feed_forward(x, act=act):
            pass
        return x

    def total_cost(self, x, y, act=sigmoid):
        y_pred = self.feed_forward(x)
        cost = self.cost(y_pred, y) + self.regularization(self.weights[-1], x.shape[1])
        return cost

    def update_mini_batch(self, mini_batch, eta, n, act=sigmoid):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        x, y = mini_batch
        d_nabla_w, d_nabla_b = self.backpropagate(x, y, act=act)

        self.weights = [
            w - eta * dnw - eta * d(w, n, fun=self.regularization)
            for w, dnw in zip(self.weights, d_nabla_w)
        ]
        self.biases = [b - eta * dnb for b, dnb in zip(self.biases, d_nabla_b)]

    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        eta,
        shuffle=True,
        evaluation_data=None,
        test_data=None,
    ):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        train_data, train_labels = training_data

        if shuffle:
            ci = np.arange(train_data.shape[1])
            np.random.shuffle(ci)
            train_data = train_data[:, ci]
            train_labels = train_labels[:, ci]

        for epoch in range(epochs):
            logger.info(f"Initiating epoch {epoch}")

            no_batches = math.ceil(train_data.shape[1] / mini_batch_size)
            data_mini_batch = np.split(train_data, no_batches, axis=1)
            labels_mini_batch = np.split(train_labels, no_batches, axis=1)

            for index, (data, labels) in enumerate(zip(data_mini_batch, labels_mini_batch)):
                self.update_mini_batch((data, labels), eta, train_data.shape[1])

                if evaluation_data:
                    accuracy = self.accuracy(evaluation_data)
                    print(f"Batch {index}/{no_batches} evaluation accuracy: {accuracy*100:2.2f}%", end='\r')
                elif test_data:
                    accuracy = self.accuracy(test_data)
                    print(f"Batch {index}/{no_batches} test accuracy: {accuracy*100:2.2f}%", end='\r')
                else:
                    print(f"Batch {index}/{no_batches} finished", end='\r')

        if test_data:
            n_test = test_data[0].shape[1]
            print(f"Final test data accuracy: {self.accuracy(test_data)}/{n_test}")

    def accuracy(self, data):
        x, y = data
        y_pred = self.feed_forward(x)

        # Compute output matrix (no softmax yet)
        matrix = np.max(y_pred, axis=0)
        y_pred = np.where(matrix == y_pred, 1, 0)

        compare = y == y_pred
        return np.sum(compare.all(axis=0)) / x.shape[1]

    def backpropagate(self, x, y, act=sigmoid):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays"""
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        w_sum, w_act = [], [x]
        for w, x in self._feed_forward(x, act=act):
            w_sum.append(w)
            w_act.append(x)

        delta = d(w_act[-1], y, fun=self.cost) * d(w_sum[-1], fun=act)
        nabla_w[-1] = np.dot(delta, w_act[-2].T) / delta.shape[1]
        nabla_b[-1] = np.average(delta, keepdims=True, axis=1)

        for i in range(self.num_layers - 2, 0, -1):
            delta = np.dot(self.weights[i].T, delta) * d(w_sum[i - 1], fun=act)
            nabla_w[i - 1] = np.dot(delta, w_act[i - 1].T) / delta.shape[1]
            nabla_b[i - 1] = np.average(delta, keepdims=True, axis=1)

        return nabla_w, nabla_b
