from data.stackoverflow.preparation import load_clients
import numpy as np


def load_clients_data(num_clients=100, seed=0, max_train_examples=20_000):
    clients = load_clients(num_clients)
    np.random.seed(seed)
    clients_ids = np.random.choice([i for i, tr in enumerate(clients) if len(tr[1]) <= max_train_examples],
                                   size=num_clients, replace=False)
    # clients_ids = np.random.randint(0, len(clients), num_clients)

    train, val, test = [], [], []
    for c_id in clients_ids:
        c = clients[c_id]
        c_len = len(c[1])
        train.append([c[0][:int(c_len * 0.6)], c[1][:int(c_len * 0.6)]])
        val.append([c[0][int(c_len * 0.6):int(c_len * 0.8)], c[1][int(c_len * 0.6):int(c_len * 0.8)]])
        test.append([c[0][int(c_len * 0.8):], c[1][int(c_len * 0.8):]])
    return train, val, test
