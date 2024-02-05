import csv

import numpy as np

from neural.utils import normalize_data, normalize_label


class MnistReader:
    def __init__(self, data_path, labels_path, delimiter=','):
        self.data_path = data_path
        self.labels_path = labels_path
        self.delimiter = delimiter

    def _data(self):
        with open(self.data_path, 'r', newline='') as csv_data:
            csv_data_reader = csv.reader(csv_data, delimiter=self.delimiter)
            for data in csv_data_reader:
                yield normalize_data(data)

    @property
    def data(self):
        return np.stack([data for data in self._data()], axis=1)

    def _labels(self):
        with open(self.labels_path, 'r', newline='') as csv_labels:
            csv_labels_reader = csv.reader(csv_labels, delimiter=self.delimiter)
            for label in csv_labels_reader:
                yield normalize_label(label)

    @property
    def labels(self):
        return np.stack([label for label in self._labels()], axis=1)

    def __iter__(self):
        for data, label in zip(self._data(), self._labels()):
            yield data, label
