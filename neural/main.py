import csv

import numpy as np

from neural import Neural, L2Regularization, cross_entropy, squared_error

class MnistReader:
    def __init__(self, data_path, labels_path, delimiter=','):
        self.data_path = data_path
        self.labels_path = labels_path
        self.delimiter = delimiter

    @staticmethod
    def _normalize_data(data):
        x = np.array(data, dtype=np.uint8) / 255
        return np.reshape(x, (x.shape[0],))

    def _data(self):
        with open(self.data_path, 'r', newline='') as csv_data:
            csv_data_reader = csv.reader(csv_data, delimiter=self.delimiter)
            for data in csv_data_reader:
                yield MnistReader._normalize_data(data)

    @property
    def data(self):
        return np.stack([data for data in self._data()], axis=1)

    @staticmethod
    def _normalize_label(label):
        y = np.zeros((10,), dtype=np.int8)
        y[int(label[0])] = 1
        return y

    def _labels(self):
        with open(self.labels_path, 'r', newline='') as csv_labels:
            csv_labels_reader = csv.reader(csv_labels, delimiter=self.delimiter)
            for label in csv_labels_reader:
                yield MnistReader._normalize_label(label)

    @property
    def labels(self):
        return np.stack([label for label in self._labels()], axis=1)

    def __iter__(self):
        for data, label in zip(self._data(), self._labels()):
            yield data, label

if __name__ == "__main__":
    training = MnistReader(
        data_path='../data/mnist_train_vectors.csv',
        labels_path='../data/mnist_train_labels.csv'
    )
    test = MnistReader(
        data_path='../data/mnist_test_vectors.csv',
        labels_path='../data/mnist_test_labels.csv',
    )

    neural = Neural([784, 100, 10], cost=cross_entropy, regularization=L2Regularization(0.4))
    neural.SGD(
        (training.data, training.labels),
        epochs=30,
        mini_batch_size=100,
        eta=0.8,
        test_data=(test.data, test.labels),
    )
