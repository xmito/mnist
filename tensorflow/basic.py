import tensorflow as tf
from tensorflow.keras import (
    datasets, callbacks, models, layers, losses, optimizers,
)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # (height, width, channels), add channels 
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    # Sequential model is useful for stacking layers where each layer has one input tensor and one output tensor
    model = models.Sequential([
        layers.Conv2D(20, 3, input_shape=(28, 28, 1), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),  # if padding is 'same' output is padded to be the size of input
        layers.Conv2D(20, 3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer="L2"),
        layers.Dense(128, activation='relu', kernel_regularizer="L2"),
        layers.Dense(10),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.85, beta_2=0.95),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    model_save_callback = callbacks.ModelCheckpoint(
        './model',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
    )
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.02, patience=4, restore_best_weights=True,
    )
    plateau = callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-7, verbose=1,
    )
    

    model.fit(
        x_train,
        y_train,
        epochs=30,
        batch_size=50,
        validation_split=0.1,  # Possible to provide validation_data tuple
        callbacks=[plateau, early_stopping, model_save_callback],
    )

    model.evaluate(x_test, y_test, verbose=2)
