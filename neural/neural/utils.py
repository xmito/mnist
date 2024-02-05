import numpy as np   

from neural.activations import sigmoid


def normalize_data(data):
    x = np.array(data, dtype=np.uint8) / 255
    return np.reshape(x, (x.shape[0],))


def normalize_label(label):
    y = np.zeros((10,), dtype=np.int8)
    y[int(label[0])] = 1
    return y


def d(*args, fun=sigmoid):
    return fun.d(*args)
