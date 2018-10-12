# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 19:55:33 2018

@author: yj-wn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
sns.set(palette="Set2")

loc_dir = {
            'our': 'proposed',
            'maddpg': 'maddpg'
           }

result = {
          'our': 0,
          'maddpg': 0,
         }

N = 60000

def readResult(root):
    rewards = []
    files = os.listdir(root)
    for f in files:
        path = os.path.join(root, f)
        with open(path, 'rb') as pickle_file:
            reward = pickle.load(pickle_file)
        reward = np.asarray(reward)
        reward = reward[:N]
        rewards.append(reward)
    return np.stack(rewards)

for mth, root in loc_dir.items():
    result[mth] = readResult(root)

##############################
##############################
##############################
# CI plot
data = {'method': [],
        'episode': [],
        'seed': [],
        'reward': [],}

for k, re in result.items():
    for seed, reward in enumerate(re):
        episode = np.arange(N).tolist()
        method = np.repeat(k, N).tolist()
        seed = np.repeat(seed, N).tolist()
        reward = reward.tolist()

        data['method'] += method
        data['seed'] += seed
        data['episode'] += episode
        data['reward'] += reward

a = pd.DataFrame(data)

plt.subplots(figsize=(10,7))
sns.lineplot(x="episode", y="reward",
             hue="method", data=a, n_boot=10, legend=False)

##############################
##############################
##############################
# Smoothing CI plot
data_smt = {'method': [],
            'episode': [],
            'seed': [],
            'reward': [],}

for k, re in result.items():
    for seed, reward in enumerate(re):
        reward = reward.tolist()
        reward = pd.DataFrame(reward).rolling(window=400).mean()
        reward = reward.dropna().values
        reward = reward.flatten()

        N = len(reward)
        episode = np.arange(N).tolist()
        method = np.repeat(k, N).tolist()
        seed = np.repeat(seed, N).tolist()
        reward = reward.tolist()

        data_smt['method'] += method
        data_smt['seed'] += seed
        data_smt['episode'] += episode
        data_smt['reward'] += reward

a_smt = pd.DataFrame(data_smt)
plt.subplots(figsize=(8,6))
sns.lineplot(x="episode", y="reward",
             hue="method", data=a_smt, n_boot=10, legend=False)
