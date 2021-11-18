from data.preparation import load_clients
import numpy as np


def load_client_datasets(num_clients=1_000):
    return load_clients('train', num_clients), load_clients('val', num_clients), load_clients('test', num_clients)


def load_clients_data(num_clients=100):
    tr, val, test = load_client_datasets(num_clients)
    return tr[:num_clients], val[:num_clients], test[:num_clients]


def filter_clients(train_clients, val_clients, test_clients, cli_num, examples_range, ret_val='clients'):
    min_examples, max_examples = examples_range[0], examples_range[1]
    print("{} Clients, size: {} - {}".format(cli_num, min_examples, max_examples))
    c_added = 0
    cls = list(range(len(train_clients)))
    cl_ds = []
    while c_added < cli_num and len(cls) > 0:
        while len(train_clients[cls[0]][0]) < min_examples or len(train_clients[cls[0]][0]) > max_examples:
            cls.pop(0)
        c_added += 1
        k = cls[0]
        cl_ds.append(k)
        cls.pop(0)
    if ret_val == 'clients':
        return [train_clients[d] for d in cl_ds],\
               [val_clients[d] for d in cl_ds],\
               [test_clients[d] for d in cl_ds]

    elif ret_val == 'total':
        x, y, v_x, v_y, t_x, t_y = [], [], [], [], [], []
        for k in cl_ds:
            x.extend(train_clients[k][0])
            y.extend(train_clients[k][1])
            v_x.extend(val_clients[k][0])
            v_y.extend(val_clients[k][1])
            t_x.extend(test_clients[k][0])
            t_y.extend(test_clients[k][1])
        x, y = np.array(x), np.array(y)
        v_x, v_y = np.array(v_x), np.array(v_y)
        t_x, t_y = np.array(t_x), np.array(t_y)
        return x, y, v_x, v_y, t_x, t_y


def filtered_clients(num_clients, examples_range=(0, 900000), ret_val='clients'):
    train_clients, val_clients, test_clients = load_client_datasets(num_clients)
    return filter_clients(train_clients, val_clients, test_clients, num_clients, examples_range, ret_val)
