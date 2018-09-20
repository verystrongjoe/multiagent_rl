import sys
import subprocess
import itertools
import sys
import subprocess
import itertools
import numpy as np


keys = ['--exp-name']
values = [['test'+ str(i) for i in range(10)]]

params = {}

# op = lambda k, l: np.concatenate([[k, v] if v is not None else [k] for v in l])
op = lambda k, l: [[k, v] if v is not None else [k] for v in l]

for s in range(len(keys)):
    params[keys[s]] = values[s]

for s in range(len(keys)):
    # key
    for k_l in itertools.combinations(keys, s+1):
        ll = []
        for k in k_l:
            ll.append(op(k, params[k]))

        for combination in itertools.product(*ll):
            if len(combination) != 0:
                print(np.concatenate(combination).tolist())
                # --scenario simple_spread --batch-size 128 --num-episodes 25000 --exp-name test --save-rate 1
                subargs = [sys.executable, 'train.py', '--scenario', 'simple_spread', '--batch-size', '128',
                           '--num-episodes', '25000', '--save-rate', '1'] + np.concatenate(combination).tolist()
                subprocess.call(subargs, shell=True)
