from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from rl2.model.ac_network import ActorNetwork, CriticNetwork
from rl2.agent.ddpg import Trainer
import numpy as np
import torch
import time
from rl2 import arglist
import pickle
from rl2.replay_buffer import SequentialMemory, MemoryBuffer
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
import experiments.timer as timer
import argparse

def parse_args():

    # save_rate = 1
    # exp_name = 'proposed_'

    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple-spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    # parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    # parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    # parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    # parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()


t_run = timer.Timer()
t_1 = timer.Timer()
t_2 = timer.Timer()
t_3 = timer.Timer()
t_4 = timer.Timer()
t_5 = timer.Timer()
t_6 = timer.Timer()
t_7 = timer.Timer()
t_8 = timer.Timer()
t_9 = timer.Timer()

t_3_1 = timer.Timer()
t_3_2 = timer.Timer()
t_3_3 = timer.Timer()
t_3_4 = timer.Timer()

t_6_1 = timer.Timer()
t_6_2 = timer.Timer()
t_6_3 = timer.Timer()
t_6_4 = timer.Timer()
t_6_5 = timer.Timer()

def runtime(f):
    def wrapper(*args, **kwargs):
        import timeit
        start = timeit.default_timer()
        v = f(*args, **kwargs)
        end = timeit.default_timer()
        print('executed time  of f {} : {}'.format(f, round(end - start, 4)))
        # return f(*a, **kwargs)
        return v
    return wrapper

@runtime
def run(arglist):

    t_1.tic()
    # load scenario from script
    scenario_name = arglist.scenario
    scenario = scenarios.load(scenario_name + ".py").Scenario()

    # create world
    world = scenario.make_world()

    # create multiagent environment
    env = MultiAgentEnv(scenario_name, world, scenario.reset_world, scenario.reward, scenario.observation)
    print('observation shape: ', env.observation_space)
    print('action shape: ', env.action_space)
    env.discrete_action_input = True
    env.discrete_action_space = False

    actor = ActorNetwork(input_dim=18, out_dim=5)
    critic = CriticNetwork(input_dim=18 + 5, out_dim=1)
    memory = MemoryBuffer(size=1000000)
    agent = Trainer(actor, critic, memory)

    # def run():
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    terminal_reward = []

    # history = []
    # history_rewards = []
    # episode_rewards = []  # sum of rewards for all agents
    episode_loss = []
    obs = env.reset()
    episode_step = 0
    train_step = 0
    nb_episode = 0

    verbose_step = False
    verbose_episode = True
    t_start = time.time()
    t_1.toc()

    print('Starting iterations...')
    while True:
        t_2.tic()
        # get action
        obs = agent.process_obs(obs)
        t_2.toc()

        t_3.tic() #todo: tuning
        actions = agent.get_exploration_action(obs, t_3_1, t_3_2, t_3_3, t_3_4)
        t_3.toc()

        t_4.tic()
        actions = agent.process_action(actions)
        t_4.toc()

        # environment step
        t_5.tic() #todo: tuning
        new_obs, rewards, done, info = env.step(actions)
        t_5.toc()
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

        for i, rew in enumerate([rewards] * env.n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        # # for displaying learned policies
        # if arglist.display:
        #     if done or terminal:
        #         time.sleep(0.1)
        #         env.render()
        #     # continue

        if done or terminal:
            obs = env.reset()
            episode_step = 0
            nb_episode += 1
            episode_rewards.append(0)
            terminal_reward.append(np.mean(rewards))

        # increment global step counter
        train_step += 1

        t_6.tic() #todo: tuning
        # update all trainers, if not in display or benchmark mode
        loss = [np.nan, np.nan]
        if train_step > arglist.warmup_steps and episode_step % 100 == 0:
            loss = agent.optimize(t_6_1, t_6_2,t_6_3,t_6_4,t_6_5)
            t_7.tic()
            loss = [loss[0].data.item(), loss[1].data.item()]
            t_7.toc()
        t_6.toc()

        episode_loss.append(loss)

        if verbose_step:
            if loss == [np.nan, np.nan]:
                loss = ['--', '--']
            print('step: {}, actor_loss: {}, critic_loss: {}'.format(train_step, loss[0], loss[1]))

        elif verbose_episode:
            if (done or terminal) and (len(episode_rewards) % arglist.save_rate == 0):
                print("steps: {}, episodes: {}, mean episode reward: {}, reward: {}, time: {}".format(
                    train_step, len(episode_rewards), round(np.mean(episode_rewards[-arglist.save_rate-1:-1]), 3),
                    round(np.mean(terminal_reward), 3), round(time.time() - t_start, 3)))
                terminal_reward = []
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate-1:-1]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

        # saves final episode reward for plotting training curve later
        if nb_episode > arglist.num_episodes:
            np.save('iter_{}_episode_rewards.npy'.format(cnt), episode_rewards)

            rew_file_name = '' + arglist.exp_name + '{}_rewards.pkl'.format(cnt)
            with open(rew_file_name, 'wb') as fp:
                pickle.dump(final_ep_rewards, fp)
            agrew_file_name = '' + arglist.exp_name + '{}_agrewards.pkl'.format(cnt)
            with open(agrew_file_name, 'wb') as fp:
                pickle.dump(final_ep_ag_rewards, fp)
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))

            break

    # np.save('history_rewards_{}.npy'.format(cnt), history_rewards)
    # np.save('history_{}.npy'.format(cnt), history)


if __name__ == '__main__':
    for cnt in range(1):
        torch.cuda.empty_cache()
        torch.set_default_tensor_type('torch.FloatTensor')
        t_run.tic()

        arglist = parse_args()
        arglist.learning_rate = 1e-3
        arglist.tau = 0.001
        arglist.warmup_steps = 200
        print(arglist.display)

        run(arglist)
        t_run.toc()
        print('run time : {}'.format(t_run.total_time))
        print('{} + {} + {} + {} + {} + {} + {} + {} + {}= {}'.format(t_1.total_time,
                                              t_2.total_time,
                                              t_3.total_time,
                                              t_4.total_time,
                                              t_5.total_time,
                                              t_6.total_time,
                                              t_7.total_time,
                                              t_8.total_time,
                                              t_9.total_time,
                                              (t_1.total_time +
                                                               t_2.total_time +
                                                               t_3.total_time +
                                                               t_4.total_time +
                                               t_5.total_time +
                                               t_6.total_time +
                                               t_7.total_time +
                                               t_8.total_time +
                                               t_9.total_time
                                               )))

        print('{} + {} + {} + {} = {}'.format(t_3_1.total_time, t_3_2.total_time, t_3_3.total_time, t_3_4.total_time,
                                              t_3_1.total_time + t_3_2.total_time + t_3_3.total_time + t_3_4.total_time))

        print('{} + {} + {} + {} + {} = {}'.format(t_6_1.total_time, t_6_2.total_time, t_6_3.total_time, t_6_4.total_time, t_6_5.total_time,
                                                  t_6_1.total_time + t_6_2.total_time + t_6_3.total_time + t_6_4.total_time + t_6_5.total_time))

##############################
'''
import numpy as np
from matplotlib import pyplot as plt
history_rewards = np.load('experiments/history_rewards.npy')

history_rewards.shape
plt.plot(history_rewards)
plt.show()

r = []
for i in range(len(history_rewards)//1000):
    x = history_rewards[(i * 1000):(i * 1000)+1000]
    r.append(np.mean(x))

r = np.array(r)
plt.plot(r)
plt.show()
'''