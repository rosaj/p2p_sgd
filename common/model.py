import tensorflow as tf
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import KLDivergence

from tensorflow.keras.layers import Embedding, LSTM, Dense, GRU, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.models import clone_model

from tensorflow.keras.callbacks import ModelCheckpoint

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

import numpy as np
import gc


class MaskedSparseCategoricalAccuracy(SparseCategoricalAccuracy):
    """An accuracy metric that masks some tokens."""

    # noinspection PyDefaultArgument
    def __init__(self, masked_tokens=[1], name='accuracy_no_oov', dtype=None):
        self._masked_tokens = masked_tokens or []
        super().__init__(name, dtype=dtype)

    @staticmethod
    def _get_mask(y_true, sample_weight, masked_tokens):
        if sample_weight is None:
            sample_weight = tf.ones_like(y_true, tf.float32)
        for token in masked_tokens:
            mask = tf.cast(tf.not_equal(y_true, token), tf.float32)
            sample_weight = sample_weight * mask
        return sample_weight

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = MaskedSparseCategoricalAccuracy._get_mask(y_true, sample_weight, self._masked_tokens)
        num_classes = tf.shape(y_pred)[-1]
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, num_classes])
        mask = tf.reshape(mask, [-1])
        super().update_state(y_true, y_pred, mask)


def compile_model(model, lr=0.01, decay=0):
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=Adam(learning_rate=lr, decay=decay),
                  metrics=[MaskedSparseCategoricalAccuracy()])


def create_keras_model(model_v=1, lr=0.01, decay=0, vocab_size=10002, embedding_size=10, do_compile=True):
    if model_v == 1:
        model = Sequential(

            [
                Embedding(vocab_size, 10, input_length=embedding_size, trainable=True, mask_zero=True),
                # Flatten(),
                LSTM(10),
                Dense(10),
                Dense(vocab_size, activation='softmax')
            ],
            name="model_1"
        )
    elif model_v == 2:
        model = Sequential(
            [
                Embedding(vocab_size, 100, input_length=embedding_size, trainable=True, mask_zero=True),
                LSTM(100),
                Dense(100),
                Dense(vocab_size, activation='softmax')
            ],
            name="model_2"
        )
    else:
        model = Sequential(
            [
                Embedding(vocab_size, 256, input_length=embedding_size, trainable=True, mask_zero=True),
                LSTM(256),
                Dense(256),
                Dense(vocab_size, activation='softmax')
            ],
            name="model_3"
        )

    if do_compile:
        compile_model(model, lr, decay)
    return model


def calculate_memory_model_size(model):
    return model.count_params() * 4 / (1024 ** 2) * 3


def load(model_path):
    if not model_path.endswith('.h5'):
        model_path += '.h5'
    # print('loading', model_path)
    model = load_model('models/' + model_path,
                       custom_objects={"MaskedSparseCategoricalAccuracy": MaskedSparseCategoricalAccuracy()})
    return model


def save(model, model_path, signatures=None):
    if not model_path.endswith('.h5'):
        model_path += '.h5'
    # print('saving', model_path)
    save_model(model, 'models/' + model_path, save_format='h5', signatures=signatures)
    clear_session()
    # gc.collect()
    # del model


def clear_session():
    tf.keras.backend.clear_session()
    gc.collect()


def load_tf(model_path):
    return tf.saved_model.load('models/' + model_path)


def save_tf(model, model_path, signatures=None):
    tf.saved_model.save(model, 'models/' + model_path, signatures=signatures)


def average_weights(model_weights):
    return __apply_on_weights(model_weights, np.average)


def __apply_on_weights(model_weights, np_func):
    weights = []

    # determine how many layers need to be averaged
    n_layers = len(model_weights[0])
    for layer in range(n_layers):
        # collect this layer from each model
        layer_weights = np.array([m_weight[layer] for m_weight in model_weights])
        # weighted average of weights for this layer
        avg_layer_weights = np_func(layer_weights, axis=0)

        weights.append(avg_layer_weights)

    return weights


@tf.function
def kl_loss_compute(logits1, logits2):
    """ KL loss
    """
    pred1 = tf.math.softmax(logits1)
    pred2 = tf.math.softmax(logits2)
    loss = tf.math.reduce_mean(tf.math.reduce_sum(pred2 * tf.math.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))

    return loss


def multiply_weights_with_num(weights, num):
    for i in range(len(weights)):
        weights[i] *= num
    return weights


def add_weights(model_weights):
    return __apply_on_weights(model_weights, np.sum)


"""
def avg_weights(weight_list):
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*weight_list):
        layer_mean = tf.math.reduce_mean(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad
"""


def average_trainable_variables(trainable_variable_list):
    n_vars = len(trainable_variable_list[0])
    train_vars = []
    for n in range(n_vars):
        n_var_list = np.array([tra_var[n].numpy() for tra_var in trainable_variable_list])
        avg_n_var = np.mean(n_var_list, axis=0)
        train_vars.append(avg_n_var)
    return train_vars


if __name__ == '__main__':
    m1 = create_keras_model()
    m2 = create_keras_model()
