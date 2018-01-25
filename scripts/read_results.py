#!/usr/bin/env python
import pickle as pkl
from pprint import pprint

# print out accuracies from some result files
for dset in ['train1', 'validation1', 'validation2']:
    for net in ['cnn-lstm', 'rn', 'lstm']:
        fname = 'data/results/result_{}_{}.pkl'.format(dset, net)
        with open(fname, 'rb') as f:
            dat = pkl.load(f)
            print(fname)
            print(dat['args'].checkpoint_dir)
            pprint(dat['acc'])
