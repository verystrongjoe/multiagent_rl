import pickle
import matplotlib.pyplot as plt
import numpy as np
save_rate = 1000

for i in range(5):

    with open('test-proposed-'+str(i)+'0_rewards.pkl', 'rb') as pickle_file:
        episode_rewards = pickle.load(pickle_file)
        len_episode_rewards = len(episode_rewards)
        print('len of episode rewards : {}'.format(len_episode_rewards))
        mean_episode_rewards = []
        for i, r in enumerate(episode_rewards):
            if i < save_rate:
                continue
            else:
                mean_episode_rewards.append(np.mean(episode_rewards[i-save_rate:i]))
        plt.plot(mean_episode_rewards)


plt.show()