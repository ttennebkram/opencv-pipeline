#!/usr/bin/env python3
"""
intro_cnn_mnist.py
A minimal "hello world" CNN for classifying MNIST digits with Keras.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():

    # 1. Load MNIST dataset (handwritten digits 0–9)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # x_train: (60000, 28, 28), uint8
    # y_train: (60000,), labels 0–9


    # 2. Normalize input to [0,1] and add channel dimension
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add a last dimension for "channels" → (N, 28, 28, 1)
    x_train = x_train[..., None]
    x_test = x_test[..., None]


    # 3. Define a tiny CNN model
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),          # H, W, C
            layers.Conv2D(8, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(10, activation="softmax"),   # 10 classes
        ]
    )


    # 4. Compile model: choose optimizer, loss, and metric
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Print a summary so you can see the layer shapes
    model.summary()


    # 5. Train the model
    model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=3,            # small number just to keep it quick
        validation_split=0.1
    )


    # 6. Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}, test loss: {test_loss:.4f}")


    # 7. Run a single prediction for demonstration
    #    (Take the first test sample.)
    sample = x_test[0:1]        # shape (1, 28, 28, 1)
    true_label = int(y_test[0])

    probs = model.predict(sample, verbose=0)[0]   # shape (10,)
    pred_label = int(probs.argmax())

    print("\nExample prediction:")
    print("  True label:     ", true_label)
    print("  Predicted label:", pred_label)
    print("  Class probs:    ", probs)


if __name__ == "__main__":
    main()


