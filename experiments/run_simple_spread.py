from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from rl2.model.ac_network import ActorNetwork, CriticNetwork
from rl2.agent.ddpg2 import Trainer
import numpy as np
import torch
import time
from rl2 import arglist
import pickle
from rl2.replay_buffer import SequentialMemory, MemoryBuffer




# load scenario from script
scenario_name = 'simple_spread'
scenario = scenarios.load(scenario_name + ".py").Scenario()

# create world
world = scenario.make_world()

# create multiagent environment
env = MultiAgentEnv(scenario_name, world, scenario.reset_world, scenario.reward, scenario.observation)

print('observation shape: ', env.observation_space)
print('action shape: ', env.action_space)
env.discrete_action_input = True
env.discrete_action_space = False

import os
import torch as th
device = th.device("cuda" if th.cuda.is_available() else "cpu")  # if gpu is to be used


th.backends.cudnn.benchmark=True
os.environ["CUDA_VISIBLE_DEVICES"] = '0'



actor = ActorNetwork(input_dim=18, out_dim=5).to(device)
critic = CriticNetwork(input_dim=18 + 5, out_dim=1).to(device)
memory = MemoryBuffer(size=1000000)
agent = Trainer(actor, critic, memory)

# def run():
history = []
history_rewards = []
episode_rewards = [0.0]  # sum of rewards for all agents
episode_loss = []
obs = env.reset()
episode_step = 0
train_step = 0
nb_episode = 0

verbose_step = False
verbose_episode = True

t_start = time.time()

print('Starting iterations...')
while True:
    # get action
    obs = agent.process_obs(obs)
    actions = agent.get_exploration_action(obs)
    actions = agent.process_action(actions)

    # environment step
    new_obs, rewards, done, info = env.step(actions)
    rewards = agent.process_reward(rewards)
    rewards = np.mean(rewards)
    episode_step += 1
    done = all(done)
    terminal = (episode_step >= arglist.max_episode_len)

    # collect experience
    # obs, actions, rewards, new_obs, done
    agent.memory.add(obs, actions, rewards, new_obs, done or terminal)
    obs = new_obs
    # episode_rewards.append(rewards)
    episode_rewards[-1] += rewards

    # for displaying learned policies
    if arglist.display:
        if done or terminal:
            time.sleep(0.1)
            env.render()
        # continue

    if done or terminal:
        obs = env.reset()
        episode_step = 0
        nb_episode += 1
        episode_rewards.append(0.0)

    # increment global step counter
    train_step += 1

    # update all trainers, if not in display or benchmark mode
    loss = [np.nan, np.nan]
    if train_step > arglist.warmup_steps:
        loss = agent.optimize()
        loss = [loss[0].data.item(), loss[1].data.item()]

    episode_loss.append(loss)

    if verbose_step:
        if loss == [np.nan, np.nan]:
            loss = ['--', '--']
        print('step: {}, actor_loss: {}, critic_loss: {}'.format(train_step, loss[0], loss[1]))

    elif verbose_episode:
        if (done or terminal) and len(episode_rewards) % arglist.save_rate == 0:
            # episode_loss = np.array(episode_loss)
            # print('episode: {episode}, step: {step}, reward: {reward:.2f}'.format(
            #     episode=nb_episode, step=train_step, reward=rewards))
            print('episode: {episode}, step: {step}, mean episode reward: {reward:.2f}, time : {time}'.format(
                episode=nb_episode, step=train_step, reward=3*np.mean(episode_rewards[-1*arglist.save_rate:][:-1]),
                time=round(time.time()-t_start, 3)
            ))
            history.append(np.nansum(episode_rewards))
            history_rewards.append(rewards)
            # episode_rewards = []
            episode_loss = []
            t_start = time.time()

    # saves final episode reward for plotting training curve later
    if nb_episode > arglist.num_episodes:
        print('...Finished total of {} episodes.'.format(len(episode_rewards)))
        break

np.save('history_rewards.npy', history_rewards)
np.save('history.npy', history)

