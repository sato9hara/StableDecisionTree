import os
import argparse
import pickle
import json
import joblib
import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys
sys.path.append('../src')
from Trees import StableTree
from TreeEvaluater import dist_tree

def eval_distance(tree0, tree1, max_depth):
    dist = []
    for depth in max_depth:
        dist.append((dist_tree(tree0, tree1, max_depth=depth, identical_threshold=True), dist_tree(tree0, tree1, max_depth=depth, identical_threshold=False)))
    return np.array(dist)

def compute_distance(Xtr, Ytr, Xte, Yte, args, eps):

    # parameters
    seed_tree = range(10)
    max_depth = range(1, 11)

    # method
    if eps < 0:
        method = 'greedy'
    else:
        method = 'eps%08d' % (int(1e+6*eps),)

    # directories
    dn = '../res/%s/' % (args.data,)
    out_dir = '../res/%s/dist/' % (args.data,)
    os.makedirs(out_dir, exist_ok=True)

    # compute distance
    dist, acc = [], []
    fn = args.data + "_%s" % (method,) + "_seed%02d_tree%02d.pkl"
    for s in range(args.start, args.end):
        
        # subsample data
        if Xtr.shape[0] > 1000:
            x, _, y, _ = train_test_split(Xtr, Ytr, train_size=1000, random_state=s)
        else:
            x, _, y, _ = train_test_split(Xtr, Ytr, train_size=0.8, random_state=s)
        xte, yte = Xte, Yte
        
        # number of data removal
        if args.n_remove < 1:
            n_remove = int(x.shape[0] * args.n_remove)
        else:
            n_remove = int(args.n_remove)
        res_dir = '%sremove_%03d/' % (dn, n_remove)
        fn = args.data + "_%s" % (method,) + "_seed%02d_tree%02d.pkl"

        # eval
        for t in seed_tree:
            with open(res_dir+fn % (s, t), 'rb') as f:
                res = pickle.load(f)
            tree0 = res['trees'][0]
            a = []
            for depth in max_depth:
                z = tree0.predict(x, max_depth=depth)
                zte = tree0.predict(xte, max_depth=depth)
                a.append((accuracy_score(y, z), accuracy_score(yte, zte)))
            acc.append(a)
            res = joblib.Parallel(n_jobs=args.n_parallel)(joblib.delayed(eval_distance)(tree0, tree1, max_depth) for tree1 in res['trees'][1:])
            dist.append(np.stack(res))
    dist = np.array(dist)
    acc = np.array(acc)
    np.savez('%sremove_%03d_%s.npz' % (out_dir, n_remove, method), dist=dist, acc=acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str)
    parser.add_argument('data', type=str)
    parser.add_argument('--n_remove', type=float, default=1)
    parser.add_argument('--n_parallel', type=int, default=-1)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=10)
    args = parser.parse_args()

    # load data list
    with open('../data/data.json', 'r') as f:
        data = json.load(f)

    # load data
    data_dir = '../data/'
    Xtr, Ytr = load_svmlight_file(data_dir + data[args.data]['train'])
    Xtr, Ytr = Xtr.toarray(), Ytr.astype(int)
    Xte, Yte = load_svmlight_file(data_dir + data[args.data]['test'])
    Xte, Yte = Xte.toarray(), Yte.astype(int)

    # test
    if args.method == 'greedy':
        compute_distance(Xtr, Ytr, Xte, Yte, args, -1)
    elif args.method == 'stable':
        for eps in np.logspace(-5, 0, 21):
            compute_distance(Xtr, Ytr, Xte, Yte, args, eps)
