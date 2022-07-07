from tensorflow import keras
import numpy as np


def load_clients_data(num_clients=100, mode='IID'):
    # Mode: IID, pathological non-IID, practical non-IID
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    if mode == 'IID':
        c_x_train = np.array_split(x_train, num_clients)
        c_y_train = np.array_split(y_train, num_clients)
        c_x_test = np.array_split(x_test, num_clients)
        c_y_test = np.array_split(y_test, num_clients)
    elif mode == 'pathological non-IID':
        # BrendanMcMahan2017
        # Non-IID, where we first sort the data by digit label,
        # divide it into 200 shards of size 300, and assign each of 100 clients 2 shards.
        # This is a pathological non-IID partition of the data, as most clients will only have examples of two digits.
        shards = 2
        shard_num = shards * num_clients
        sorted_ind = np.argsort(y_train)
        shard_ind = np.split(sorted_ind, shard_num)
        np.random.shuffle(shard_ind)

        c_x_train, c_y_train = [], []
        for i in range(0, len(shard_ind), shards):
            cxtr, cytr = [], []
            for sh_ind in range(i, i+shards):
                cxtr.extend(x_train[shard_ind][sh_ind])
                cytr.extend(y_train[shard_ind][sh_ind])
            c_x_train.append(cxtr)
            c_y_train.append(cytr)

        # Test data is uniformly distributed
        c_x_test = np.array_split(x_test, num_clients)
        c_y_test = np.array_split(y_test, num_clients)
    elif mode == 'practical non-IID':
        # Huang2020
        # A practical non-IID data
        cls, cls_count = np.unique(y_train, return_counts=True)
        n_groups, num_classes = 5, len(cls)
        group_clients = int(num_clients / n_groups)
        n_group_classes = int(num_classes / n_groups)
        sample_pct = (0.8, 0.2)
        groups = {_: [] for _ in range(int(num_classes/n_group_classes))}
        for i_n in range(0, num_classes, n_group_classes):
            g_ind = int(i_n / n_group_classes)
            domin_ind = []
            for i in range(i_n, i_n + n_group_classes):
                domin_ind.append(i)
                num_samples = int(cls_count[i] * sample_pct[0])
                inds = np.where(y_train == i)[0]
                y_samples = np.array_split(y_train[inds[:num_samples]], group_clients)
                x_samples = np.array_split(x_train[inds[:num_samples]], group_clients)
                y_train = np.delete(y_train, inds[:num_samples])
                x_train = np.delete(x_train, inds[:num_samples], axis=0)
                groups[g_ind].append((x_samples, y_samples))
            for j in np.delete(np.array(range(num_classes)), domin_ind):
                num_samples = int(cls_count[j] * sample_pct[1] / (n_groups - 1))
                inds = np.where(y_train == j)[0]
                y_samples = np.array_split(y_train[inds[:num_samples]], group_clients)
                x_samples = np.array_split(x_train[inds[:num_samples]], group_clients)
                y_train = np.delete(y_train, inds[:num_samples])
                x_train = np.delete(x_train, inds[:num_samples], axis=0)
                groups[g_ind].append((x_samples, y_samples))

        """ Sanity check
        for gk, gv in groups.items():
            print(gk, end=' ')
            for i in range(10):
                c = sum([len(d) for d in gv[i][1]])
                print(round(c/cls_count[gv[i][1][0][0]], 2), '({})'.format(gv[i][1][0][0]), end=' ')
            print()
        """
        x_clients = {_: [] for _ in range(num_clients)}
        y_clients = {_: [] for _ in range(num_clients)}
        for gk, gv in groups.items():
            for cl in gv:
                for i, ci in enumerate(list(x_clients.keys())[gk*group_clients:(gk+1)*group_clients]):
                    x_clients[ci].extend(cl[0][i])
                    y_clients[ci].extend(cl[1][i])

        c_x_train = [x_clients[_] for _ in range(num_clients)]
        c_y_train = [y_clients[_] for _ in range(num_clients)]
        # Test data is uniformly distributed
        c_x_test = np.array_split(x_test, num_clients)
        c_y_test = np.array_split(y_test, num_clients)
    else:
        raise ValueError("Invalid mode")

    c_train = list(zip(c_x_train, c_y_train))
    c_test = list(zip(c_x_test, c_y_test))
    c_val = [([], []) for _ in range(len(c_train))]
    return c_train, c_val, c_test
