import tensorflow as tf
import numpy as np
import logging
import os


def set_seed(seed):
    if seed:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        print("SEED:", seed)


def set_tf_log_level(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)
