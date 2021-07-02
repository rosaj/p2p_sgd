import tensorflow as tf
import numpy as np


def set_seed(seed):
    if seed:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        print("SEED:", seed)
