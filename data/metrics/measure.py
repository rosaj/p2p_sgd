import numpy as np
from .shannon import js_div


def convert_to_global_vector(data_ds, global_space):
    ds = []
    for c_cls in data_ds:
        f_cls = np.zeros(global_space)
        value, counts = np.unique(c_cls, return_counts=True)
        f_cls[value - 2] = counts
        ds.append(list(f_cls))
    return ds


def get_avg_distance(ds):
    shape = (len(ds), len(ds))
    ret = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(i, shape[1]):
            if i == j:
                continue
            ret[i, j] = js_div(ds[i], ds[j])

    avg_js_div = np.mean(list(ret[i] for i in zip(*np.triu_indices_from(ret, k=1))))
    return avg_js_div, ret


def multiprocess_get_avg_distance(ds):
    from multiprocessing.pool import Pool
    from multiprocessing import Lock
    import multiprocessing

    manager = multiprocessing.Manager()
    ret_dict = manager.dict()
    lock = Lock()

    shape = (len(ds), len(ds))

    def calc_js_div(i):
        for j in range(i, shape[1]):
            if i == j:
                continue
            js_res = js_div(ds[i], ds[j])
            lock.acquire()
            try:
                print(i, j, js_res)
                ret_dict["{}-{}".format(i, j)] = js_res
            except Exception as e:
                print(e)
            finally:
                lock.release()

    p = Pool(processes=len(ds))
    p.map(calc_js_div, range(shape[0]))
    p.close()
    p.join()

    ret = np.zeros(shape)
    for k, v in ret_dict.items():
        ids = k.split('-')
        ret[int(ids[0]), int(ids[1])] = v
    avg_js_div = np.mean(list(ret[i] for i in zip(*np.triu_indices_from(ret, k=1))))
    return avg_js_div, ret
