import numpy as np
from sklearn.preprocessing import LabelEncoder

from utils import predict, compute_acc, split_sort_idx, sample_from_p

class MyTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.n_nodes = 0
        self.feature = []
        self.threshold = []
        self.value = []
        self.n_node_samples = []
        self.children_left = []
        self.children_right = []
        self.labels = LabelEncoder()
        self.n_labels = -1

    def __compile(self):
        self.feature = np.array(self.feature)
        self.threshold = np.array(self.threshold)
        self.value = np.array([[-1] * self.n_labels if v is None else v for v in self.value])
        self.children_left = np.array(self.children_left)
        self.children_right = np.array(self.children_right)
        return self

    def fit(self, x, y):
        return self.__compile()

    def predict(self, x, max_depth=-1):
        y = predict(x, self.feature, self.threshold, self.value, self.children_left, self.children_right, self.n_nodes-1, max_depth)
        return self.labels.inverse_transform(y)

class StableTree(MyTree):
    def __init__(self, max_depth=3, seed=0, eps=0.1, n_thresholds=500):
        super().__init__(max_depth)
        self.max_depth = max_depth
        self.seed = seed
        self.eps = eps
        self.n_thresholds = n_thresholds
    
    def fit(self, x, y, remove=[]):
        self.labels = self.labels.fit(y)
        self.n_labels = len(self.labels.classes_)
        y = self.labels.transform(y)

        # thresholds
        thresholds = []
        for d in range(x.shape[1]):
            xd = np.linspace(x[:, d].min(), x[:, d].max(), self.n_thresholds+1)[:-1]
            thresholds.append(np.stack([np.array([d]*xd.size), xd], axis=1))
        thresholds = np.concatenate(thresholds, axis=0)

        # fit children
        sort_idx = np.stack([np.argsort(x[:, d]) for d in range(x.shape[1])], axis=1)
        counts = np.eye(self.n_labels, dtype=int)[y]
        _, acc, n = self.train_children(x, y, counts, sort_idx, thresholds, 0, self.max_depth, eps=self.eps, remove=remove)
        self.n_nodes = n + 1
        self.acc = acc / x.shape[0]
        return self._MyTree__compile()
    
    def train_children(self, x, y, counts, sort_idx, thresholds, node, depth, eps=-1, remove=[]):
        
        # node info
        idx_0 = np.setdiff1d(sort_idx[:, 0], remove)
        node_value = np.sum(counts[idx_0, :], axis=0)
        node_samples = np.sum(node_value)
        if (node_samples <= 1) or (depth == 0) or (np.sum(node_value > 0) == 1):
            self.feature.append(-2)
            self.threshold.append(-2.0)
            self.value.append(node_value)
            self.n_node_samples.append(node_samples)
            self.children_left.append(-1)
            self.children_right.append(-1)
            return np.argmax(node_value), max(node_value), len(self.feature) - 1

        # sampling
        a = compute_acc(x, y, self.n_labels, sort_idx, thresholds, np.array(remove, dtype=int))
        if eps < 0: # greedy
            p = (a[:, 2] == a[:, 2].max()).astype(float)
        else:
            p = (a[:, 2] / a[:, 2].max()) * 2 * np.log(np.sum(a[:, 2]>=0)) / eps
            p = np.exp(p - p.max())
            p[a[:, 2] < 0] = 0
        p = p / p.sum()
        j = sample_from_p(p, self.seed + node)
        
        # split info
        split_dim, split_th = int(a[j, 0]), a[j, 1]

        # split
        idx = (x[sort_idx[:, split_dim], split_dim] <= split_th)
        if np.sum(idx) == 0 or np.sum(~idx) == 0:
            self.feature.append(-2)
            self.threshold.append(-2.0)
            self.value.append(node_value)
            self.n_node_samples.append(node_samples)
            self.children_left.append(-1)
            self.children_right.append(-1)
            return np.argmax(node_value), max(node_value), len(self.feature)
        sort_idx_left = np.zeros((np.sum(idx), sort_idx.shape[1]), dtype=int)
        sort_idx_right = np.zeros((np.sum(~idx), sort_idx.shape[1]), dtype=int)
        split_sort_idx(x, split_dim, split_th, sort_idx, sort_idx_left, sort_idx_right)

        # recursion
        y_left, acc_left, n_left = self.train_children(x, y, counts, sort_idx_left, thresholds, 2*node+1, depth-1, eps=eps, remove=remove)
        y_right, acc_right, n_right = self.train_children(x, y, counts, sort_idx_right, thresholds, 2*node+2, depth-1, eps=eps, remove=remove)
        
        # pruning
        if y_left >= 0 and y_left == y_right:
            self.feature.append(-2)
            self.threshold.append(-2.0)
            self.value.append(node_value)
            self.n_node_samples.append(node_samples)
            self.children_left.append(-1)
            self.children_right.append(-1)
            return np.argmax(node_value), max(node_value), len(self.feature) - 1
        else:
            self.feature.append(split_dim)
            self.threshold.append(split_th)
            self.value.append(node_value)
            self.n_node_samples.append(node_samples)
            self.children_left.append(n_left)
            self.children_right.append(n_right)
            return -1, acc_left + acc_right, len(self.feature) - 1
