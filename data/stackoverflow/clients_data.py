from data.stackoverflow.preparation import load_clients
import numpy as np
from data.util import random_choice_with_seed


def load_client_datasets(num_clients, max_client_num=10_000, directory='clients'):
    clients = load_clients(num_clients, max_client_num=max_client_num, directory=directory)
    train, val, test, tags = [], [], [], []
    for c_id in range(len(clients)):
        c = clients[c_id]
        c_len = len(c[1])
        train.append([c[0][:int(c_len * 0.6)], c[1][:int(c_len * 0.6)]])
        val.append([c[0][int(c_len * 0.6):int(c_len * 0.8)], c[1][int(c_len * 0.6):int(c_len * 0.8)]])
        test.append([c[0][int(c_len * 0.8):], c[1][int(c_len * 0.8):]])
        tags.append([item.decode() for sublist in c[2] for item in sublist])
    return train, val, test, tags


def load_clients_data(num_clients=100, seed=608361, train_examples_range=(700, 20_000), max_client_num=10_000, directory='clients'):
    train, val, test, tags = load_client_datasets(25_000, max_client_num=max_client_num, directory=directory)
    choices = [i for i, tr in enumerate(train) if train_examples_range[0] < len(tr[1]) <= train_examples_range[1]]
    client_ids = random_choice_with_seed(choices, num_clients, seed)
    data = {
        "train": [el for ei, el in enumerate(train) if ei in client_ids],
        "val": [el for ei, el in enumerate(val) if ei in client_ids],
        "test": [el for ei, el in enumerate(test) if ei in client_ids],
        "metadata-tags": [el for ei, el in enumerate(tags) if ei in client_ids],
        "dataset_name": ['stackoverflow-nwp'] * num_clients,
    }
    return data
