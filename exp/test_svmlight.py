import os
import argparse
import pickle
import json
import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

from TreeEvaluater import eval_stability

import sys
sys.path.append('../src')
from Trees import StableTree

def test(Xtr, Ytr, Xte, Yte, args, eps):
    for seed in range(args.start, args.end):
        if eps < 0:
            method = 'greedy'
        else:
            method = 'eps%08d' % (int(1e+6*eps),)

        # subsample data
        if Xtr.shape[0] > 1000:
            x, _, y, _ = train_test_split(Xtr, Ytr, train_size=1000, random_state=seed)
        else:
            x, _, y, _ = train_test_split(Xtr, Ytr, train_size=0.8, random_state=seed)
        xte, yte = Xte, Yte
        
        # number of data removal
        if args.n_remove < 1:
            n_remove = int(x.shape[0] * args.n_remove)
        else:
            n_remove = int(args.n_remove)

        # output dir & file
        dn = '../res/%s/remove_%03d/' % (args.data, n_remove)
        os.makedirs(dn, exist_ok=True)
        fn = args.data + "_%s" % (method,) + "_seed%02d_tree%02d.pkl"

        # randomness over seed_tree
        for seed_tree in range(0, 10):
            removes, trees, acc = eval_stability(x, y, xte, yte, eps, args.max_depth, args.trial, n_remove=n_remove, seed=seed, seed_tree=seed_tree, n_thresholds=args.n_thresholds, n_parallel=args.n_parallel)
            with open(dn+fn % (seed, seed_tree), 'wb') as f:
                pickle.dump({'remove':removes, 'trees':trees, 'acc':acc}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str)
    parser.add_argument('data', type=str)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--trial', type=int, default=100)
    parser.add_argument('--n_remove', type=float, default=1)
    parser.add_argument('--n_thresholds', type=int, default=500)
    parser.add_argument('--n_parallel', type=int, default=-1)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=10)
    args = parser.parse_args()

    # load data list
    with open('../data/data.json', 'r') as f:
        data = json.load(f)

    # output dir
    dn = '../res/%s/' % (args.data,)
    os.makedirs(dn, exist_ok=True)

    # experiment
    data_dir = '../data/'
    Xtr, Ytr = load_svmlight_file(data_dir + data[args.data]['train'])
    Xtr, Ytr = Xtr.toarray(), Ytr.astype(int)
    Xte, Yte = load_svmlight_file(data_dir + data[args.data]['test'])
    Xte, Yte = Xte.toarray(), Yte.astype(int)

    # test
    if args.method == 'greedy':
        test(Xtr, Ytr, Xte, Yte, args, -1)
    elif args.method == 'stable':
        for eps in np.logspace(-5, 0, 21):
            test(Xtr, Ytr, Xte, Yte, args, eps)
