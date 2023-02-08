import numpy as np
import numba

MIN_VAL = - 1e+8

@numba.njit("int64[:](float64[:, :], int64[:], float64[:], int64[:, :], int64[:], int64[:], int64, int64)")
def predict(x, feature, threshold, value, children_left, children_right, j0, max_depth):
    y = np.zeros(x.shape[0], dtype=np.int64)
    for i in range(x.shape[0]):
        j, d = j0, 0
        while True:
            if (children_left[j] < 0 and children_right[j] < 0) or (d >= max_depth and max_depth >= 0):
                y[i] = np.argmax(value[j])
                break
            if x[i, feature[j]] <= threshold[j]:
                j = children_left[j]
            else:
                j = children_right[j]
            d = d + 1
    return y

@numba.njit("float64[:, :](float64[:, :], int64[:], int64, int64[:, :], float64[:, :], int64[:])")
def compute_acc(x, y, n_labels, sort_idx, thresholds, remove):
    ny = np.zeros(n_labels)
    for i in sort_idx[:, 0]:
        if i in remove:
            continue
        ny[y[i]] = ny[y[i]] + 1
    a = np.zeros((thresholds.shape[0], 3), dtype=np.float64)
    a[:, 2] = MIN_VAL
    d_prev = -1
    for k, (d, t) in enumerate(thresholds):
        d = int(d)
        if d != d_prev:
            n_left = np.zeros(n_labels)
            n_right = np.array([n for n in ny])
            i = 0
        d_prev = d
        while (i < sort_idx.shape[0]) and (x[sort_idx[i, d], d] <= t):
            if sort_idx[i, d] in remove:
                i = i + 1
                continue
            n_left[y[sort_idx[i, d]]] = n_left[y[sort_idx[i, d]]] + 1
            n_right[y[sort_idx[i, d]]] = n_right[y[sort_idx[i, d]]] - 1
            i = i + 1
        a[k, 0] = d
        a[k, 1] = t
        if (i == 0) or (i == sort_idx.shape[0]):
            a[k, 2] = MIN_VAL
        else:
            a[k, 2] = n_left.max() + n_right.max()
    return a

@numba.njit("void(float64[:, :], int64, float64, int64[:, :], int64[:, :], int64[:, :])", parallel=True)
def split_sort_idx(x, split_dim, split_th, sort_idx, sort_idx_left, sort_idx_right):
    for d in numba.prange(sort_idx.shape[1]):
        l, r = 0, 0
        for i in sort_idx[:, d]:
            if x[i, split_dim] <= split_th:
                sort_idx_left[l, d] = int(i)
                l = l + 1
            else:
                sort_idx_right[r, d] = int(i)
                r = r + 1

@numba.njit("int64(float64[:], int64)")
def sample_from_p(p, seed):
    np.random.seed(seed)
    while True:
        j = np.random.choice(p.size, 10000)
        t = np.random.rand(j.size)
        k = np.where(t < p[j])[0]
        if k.size == 0:
            continue
        else:
            j = j[k[0]]
            break
    return j
