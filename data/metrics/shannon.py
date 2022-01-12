from scipy.stats import entropy
from scipy.special import rel_entr
import numpy as np


def ent(l):
    return entropy(l, base=2)


# relative entropy or Kullback-Leibler Divergence
# id not a distance metric because it is not symmetric
def rel_ent(p, q, e=1e-15):
    assert len(p) == len(q)
    pn = [i / sum(p) for i in p]
    qn = [i / sum(q) for i in q]

    return sum(
        pn[i] * np.log2(np.clip(pn[i], e, 1.0) / np.clip(qn[i], e, 1.0))
        for i in range(len(pn))
    )


def js_div(p, q):
    assert len(p) == len(q)
    pn = [i / sum(p) for i in p]
    qn = [i / sum(q) for i in q]

    m = 0.5 * (np.array(pn) + np.array(qn))
    return 0.5 * rel_ent(pn, m) + 0.5 * rel_ent(qn, m)


def get_avg_distance(ds):
    shape = (len(ds), len(ds))
    ret = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(i, shape[1]):
            if i == j:
                continue
            ret[i, j] = js_div(ds[i], ds[j])

    avg_js_div = np.mean(list(ret[i] for i in zip(*np.triu_indices_from(ret, k=1))))
    return avg_js_div


def convert_to_global_vector(data_ds, global_space):
    ds = []
    for c_cls in data_ds:
        f_cls = np.zeros(global_space)
        value, counts = np.unique(c_cls, return_counts=True)
        f_cls[value - 2] = counts
        ds.append(list(f_cls))
    return ds
