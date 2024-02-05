import numpy as np
import tensorflow as tf
from tensorflow.keras import (
    layers, datasets, losses, optimizers, metrics, callbacks, regularizers,
)
from tensorflow.keras import Model


class MyModel(Model):
  def __init__(self):
    super().__init__()
    self.conv1 = layers.Conv2D(20, 3, input_shape=(28, 28, 1), activation='relu')
    self.max1 = layers.MaxPooling2D(pool_size=(2, 2), padding='valid')  # if padding is 'same' output is padded to be the size of input
    self.conv2 = layers.Conv2D(20, 3, activation='relu')
    self.max2 = layers.MaxPooling2D(pool_size=(2, 2), padding='valid')  # if padding is 'same' output is padded to be the size of input
    self.flatten = layers.Flatten()
    self.dense1 = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.L2(0.1))
    self.dense2 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.L2(0.1))
    self.dense3 = layers.Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.max1(x)
    x = self.conv2(x)
    x = self.max2(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.dense3(x)
    return x


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add channels dimension (3 channels for RGB, 1 for grayscale)
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    train_samples = int(x_train.shape[0] * 0.9)
    x_train, x_val = np.split(x_train, [train_samples])
    y_train, y_val = np.split(y_train, [train_samples])

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    ).shuffle(10000).batch(100)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(100)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)

    model = MyModel()

    loss_object = losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.85, beta_2=0.95)

    train_loss = metrics.Mean(name='train_loss')
    train_accuracy = metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = metrics.Mean(name='validation loss')
    val_accuracy = metrics.SparseCategoricalAccuracy(name='test_accuracy')

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
    def validation_step(images, labels):
        predictions = model(images, training=False)
        v_loss = loss_object(labels, predictions)

        val_loss(v_loss)
        val_accuracy(labels, predictions)


    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.02, patience=4, restore_best_weights=True,
    )
    early_stopping.set_model(model)
    early_stopping.on_train_begin()

    EPOCHS = 20
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            validation_step(test_images, test_labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Validation Loss: {val_loss.result()}, '
            f'Validation Accuracy: {val_accuracy.result() * 100}'
        )

        early_stopping.on_epoch_end(epoch + 1, logs={"val_loss": val_loss.result()})
        if model.stop_training:
            model.save('./model')
            break

    test_loss = metrics.Mean(name='test_loss')
    test_accuracy = metrics.SparseCategoricalAccuracy(name='test_accuracy')

    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    print(
        f'Test Loss: {test_loss.result()}',
        f'Test Accuracy: {test_accuracy.result() * 100}',
    )
