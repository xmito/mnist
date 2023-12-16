import tensorflow as tf
from tensorflow.keras import datasets, callbacks, models, layers, losses

print("TensorFlow version:", tf.__version__)


class myCallback(callbacks.Callback):
    # Callback can be registered in fit, evaluate and predict model calls
    def on_epoch_begin(self, epoch, logs=None):
        pass
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('accuracy') >= 0.6): # Experiment with changing this value
            print("\nReached 60% accuracy")
            # self.model.stop_training = True
    # Also it is possible to access self.model 
    # self.model.stop_training = True
    # or change learning rate on optimizer
    # self.model.optimizer.learning_rate
    # Save model at certain periods
    # Extract visualizations of intermediate features at the end of each epoch


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # For grayscale images, the channels dimension is typically the third dimension.
    # The dimensions are often represented in the order (height, width, channels),
    # where the channels dimension has a size of 1 for grayscale images.
    # For RGB model, there would be 3 channels
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    # Sequential model is useful for stacking layers where each layer has one input tensor and one output tensor
    model = models.Sequential([
        layers.Conv2D(20, (5, 5), input_shape=(28, 28, 1), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),  # if padding is 'same' output is padded to be the size of input
        layers.Conv2D(20, (5, 5), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
        layers.Flatten(),
        layers.Dense(1000, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(10),
    ])

    loss = losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy'],
    )

    model_save_callback = callbacks.ModelCheckpoint(
        './mnist_model',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
    )
    callback = myCallback()

    model.fit(
        x_train,
        y_train,
        epochs=12,
        batch_size=32,
        validation_split=0.1,  # Possible to provide validation_data tuple
        callbacks=[model_save_callback, callback],
    )
    model.evaluate(x_test, y_test, verbose=2)



    # The same model with softmax activation
    probability_model = models.Sequential([
        model,
        layers.Softmax(),
    ])
    output = probability_model(x_test[:1])
    predictions = model(x_test[:1])
    test_loss = loss(y_test[:1], predictions)
    

# A tf.Tensor is immutable. You can't change a tensor once it's created. It has a value,
# but no state. All the operations discussed so far are also stateless: the output of a
# tf.matmul only depends on its inputs.
# A tf.Variable has internal state—its value. When you use the variable, the state is read.
# It's normal to calculate a gradient with respect to a variable, but the variable's state
# blocks gradient calculations from going farther back. For example: