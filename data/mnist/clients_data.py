from tensorflow import keras
import numpy as np


def load_clients_data(num_clients=100, mode='IID'):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    c_train = np.array_split(x_train, num_clients)
    c_test = np.array_split(x_test, num_clients)
    return c_train, c_train, c_test
