from common import abstract_model as ab_mod

from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.optimizers import Adam


def create_model(model_v=1, lr=0.001, decay=0, num_classes=10, input_shape=(28, 28, 1), do_compile=True,
                 default_weights=True):
    if model_v == 1:
        model = Sequential(
            [
                Input(shape=input_shape),
                Conv2D(32, kernel_size=(3, 3), activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(64, kernel_size=(3, 3), activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dropout(0.5),
                Dense(num_classes, activation="softmax"),
            ], "model_{}_{}".format(model_v, ab_mod.next_model_id('mnist')))
    elif model_v == 2:
        model = Sequential(
            [
                Input(shape=input_shape),
                Conv2D(32, kernel_size=(3, 3), activation="relu"),
                BatchNormalization(momentum=0.9, scale=False, center=False),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(64, kernel_size=(3, 3), activation="relu"),
                BatchNormalization(momentum=0.9, scale=False, center=False),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dropout(0.5),
                Dense(num_classes, activation="softmax"),
            ], "model_{}_{}".format(model_v, ab_mod.next_model_id('mnist')))
    else:
        raise ValueError("Invalid model version")

    if do_compile:
        compile_model(model, lr, decay)

    if default_weights:
        ab_mod.assign_default_weights(model, 'mnist' + str(model_v))

    return model


def compile_model(model, lr=0.001, decay=0):
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=Adam(learning_rate=lr, decay=decay),
                  metrics=[SparseCategoricalAccuracy()])


def load(model_path):
    return ab_mod.load('mnist/' + model_path, )


def save(model, model_path):
    ab_mod.save(model, 'mnist/' + model_path)
