import tensorflow as tf
from tensorflow.keras import layers, datasets, losses, optimizers, metrics
from tensorflow.keras import Model


class MyModel(Model):
  def __init__(self):
    super().__init__()
    self.conv1 = layers.Conv2D(32, 3, activation='relu')
    self.flatten = layers.Flatten()
    self.d1 = layers.Dense(128, activation='relu')
    self.d2 = layers.Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # For grayscale images, the channels dimension is typically the third dimension.
    # The dimensions are often represented in the order (height, width, channels),
    # where the channels dimension has a size of 1 for grayscale images.
    # For RGB model, there would be 3 channels
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    ).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    model = MyModel()
    loss_object = losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = optimizers.Adam()

    train_loss = metrics.Mean(name='train_loss')
    train_accuracy = metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = metrics.Mean(name='test_loss')
    test_accuracy = metrics.SparseCategoricalAccuracy(name='test_accuracy')


    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)


    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 5
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
