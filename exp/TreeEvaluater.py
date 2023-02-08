import numpy as np
import joblib
from sklearn.metrics import accuracy_score

import sys
sys.path.append('../src')
from Trees import StableTree

def is_leaf(tree, i):
    return (tree.children_left[i] < 0) and (tree.children_right[i] < 0)

def is_max_depth(current_depth, max_depth):
    return current_depth >= max_depth

def count_node(tree, i, current_depth, max_depth):
    if is_max_depth(current_depth, max_depth):
        return 1
    if is_leaf(tree, i):
        return 2**(max_depth - current_depth+1) - 1
    c_left = count_node(tree, tree.children_left[i], current_depth+1, max_depth)
    c_right = count_node(tree, tree.children_right[i], current_depth+1, max_depth)
    return c_left + c_right + 1

def dist_tree(tree1, tree2, i=-1, current_depth=0, max_depth=-1, identical_threshold=True):
    if max_depth < 0:
        max_depth = max(tree1.max_depth, tree2.max_depth)
    else:
        max_depth = min(max_depth, max(tree1.max_depth, tree2.max_depth))
    if (is_leaf(tree1, i) and is_leaf(tree2, i)) or is_max_depth(current_depth, max_depth):
        y1 = -2 if tree1.value[i] is None else np.argmax(tree1.value[i])
        y2 = -2 if tree2.value[i] is None else np.argmax(tree2.value[i])
        c1, c2 = 0, 0
        if is_leaf(tree1, i) or is_leaf(tree2, i):
            c1 = count_node(tree1, i, current_depth, max_depth)
            c2 = count_node(tree2, i, current_depth, max_depth)
        return 0 if y1 == y2 else c1+c2
    flag = (is_leaf(tree1, i) ^ is_leaf(tree2, i))
    flag = flag or (tree1.feature[i] != tree2.feature[i])
    if identical_threshold:
        flag = flag or (tree1.threshold[i] != tree2.threshold[i])
    if flag:
        c1 = count_node(tree1, i, current_depth, max_depth)
        c2 = count_node(tree2, i, current_depth, max_depth)
        return c1 + c2
    dist_left = dist_tree(tree1, tree2, 2 * i + 1, current_depth+1, max_depth, identical_threshold)
    dist_right = dist_tree(tree1, tree2, 2 * i + 2, current_depth+1, max_depth, identical_threshold)
    return dist_left + dist_right

def fit_tree(x, y, xte, yte, tree_fn, remove):
    x, y, xte, yte = x.copy(), y.copy(), xte.copy(), yte.copy()
    tree = tree_fn().fit(x, y, remove=remove)
    idx = np.delete(np.arange(x.shape[0]), remove)
    a = accuracy_score(tree.predict(x[idx]), y[idx])
    ate = accuracy_score(tree.predict(xte), yte)
    return remove, tree, a, ate

def eval_stability(x, y, xte, yte, eps, max_depth, trial, n_remove=1, seed=0, seed_tree=1, n_thresholds=1.0, n_parallel=-1):
    tree_fn = lambda: StableTree(max_depth=max_depth, seed=seed_tree, eps=eps, n_thresholds=n_thresholds)

    # evaluation by random removal
    np.random.seed(seed)
    num = x.shape[0]
    data_idx = np.arange(num)
    target_idx = [[]]
    if n_remove == 1:
        if trial > num:
            target_idx = target_idx + [[n] for n in data_idx]
        else:
            target_idx = target_idx + [[n] for n in np.random.choice(num, trial, replace=False)]
    else:
        target_idx = target_idx + [np.random.choice(num, n_remove, replace=False).tolist() for _ in range(trial)]
    fit_tree_fn = lambda remove: fit_tree(x, y, xte, yte, tree_fn, remove)
    res = joblib.Parallel(n_jobs=n_parallel)(joblib.delayed(fit_tree_fn)(remove) for remove in target_idx)

    # results
    removes = [[]]
    trees = [r[1] for r in res if len(r[0])==0]
    accs = [(r[2], r[3]) for r in res if len(r[0])==0]
    for r in res:
        if len(r[0]) == 0:
            continue
        removes.append(r[0])
        trees.append(r[1])
        accs.append((r[2], r[3]))
    return removes, trees, np.array(accs)

def eval_accuracy(x, y, xte, yte, eps, max_depth, seed_tree=1, sample_fn='p'):
    tree_fn = lambda: StableTree(max_depth=max_depth, seed=seed_tree, eps=eps, sample_fn=sample_fn)
    tree = tree_fn().fit(x, y)
    acc = []
    for d in range(tree.max_depth+1):
        a = accuracy_score(tree.predict(x, max_depth=d), y)
        ate = accuracy_score(tree.predict(xte, max_depth=d), yte)
        acc.append((a, ate))
    return np.array(acc)
