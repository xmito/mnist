from neural import Neural
from neural.errors import cross_entropy
from neural.mnist_reader import MnistReader
from neural.regularizations import L2Regularization


if __name__ == "__main__":
    training = MnistReader(
        data_path='../data/mnist_train_vectors.csv',
        labels_path='../data/mnist_train_labels.csv'
    )
    test = MnistReader(
        data_path='../data/mnist_test_vectors.csv',
        labels_path='../data/mnist_test_labels.csv',
    )

    neural = Neural(
        [784, 100, 10],
        cost=cross_entropy,
        regularization=L2Regularization(0.4)
    )
    neural.SGD(
        (training.data, training.labels),
        epochs=30,
        mini_batch_size=100,
        eta=0.8,
        test_data=(test.data, test.labels),
    )
