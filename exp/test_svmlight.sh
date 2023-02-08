#!/bin/bash

# 'covtype'
# 'webspam'
# 'diabetes'
# 'breast_cancer'
# 'ijcnn'
# 'cod-rna'
# 'diabetes'
# 'sensorless'
# 'higgs'

# evaluate average sensitivity as well as accuracy for 
for r in 1 0.010 0.030 0.10 0.30; do
    for i in "${@}"; do
        python test_svmlight.py greedy $i --max_depth 10 --trial 100 --n_remove $r --n_thresholds 500 --n_parallel -1 --start 0 --end 10 &&
        python test_svmlight.py stable $i --max_depth 10 --trial 100 --n_remove $r --n_thresholds 500 --n_parallel -1 --start 0 --end 10 &&
        python eval_svmlight.py greedy $i --n_remove $r --n_parallel -1 --start 0 --end 10 &&
        python eval_svmlight.py stable $i --n_remove $r --n_parallel -1 --start 0 --end 10
    done
done
