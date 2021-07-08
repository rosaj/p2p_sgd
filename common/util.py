import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import psutil

import tensorflow as tf
import numpy as np


def set_seed(seed):
    if seed:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        print("SEED:", seed)


def available_device_memory(device_name):
    if 'GPU' in device_name.upper():
        usage = tf.config.experimental.get_memory_info(device_name)
        return 10760 - round(usage['current'] / (1024 ** 2), 2)
    elif 'CPU' in device_name.upper():
        return psutil.virtual_memory()[1] / 1024**2


def gpu_memory(gpu_name):
    usage = tf.config.experimental.get_memory_info(gpu_name)
    return {'current (GB)': round(usage['current'] / (1024 ** 3), 2),
            'peak (GB)': round(usage['peak'] / (1024 * 1024 ** 3), 2)}


def memory_info():
    gpu_devices = tf.config.list_physical_devices('GPU')
    mem_dict = {}
    for device in gpu_devices:
        dev_name = device.name.replace('/physical_device:', '')
        mem_dict[dev_name] = str(gpu_memory(dev_name)['current (GB)']) + '/10.7 GB'
    ram_mem = psutil.virtual_memory()
    mem_dict['RAM'] = "{}/{} GB".format(round((ram_mem[0]-ram_mem[1]) / 1024**3, 2), round(ram_mem[0] / 1024**3, 2))
    return mem_dict


"""
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
"""
