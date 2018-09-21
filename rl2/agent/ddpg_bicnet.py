# reference: https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/train.py
from __future__ import division
import torch
import torch.nn.functional as F
import numpy as np
import shutil
from rl2 import arglist
import copy
from rl2.utils import to_categorical

arglist.batch_size = 128
GAMMA = 0.95
TAU = 0.001

import torch as th
device = th.device("cuda" if th.cuda.is_available() else "cpu")  # if gpu is to be used

class Trainer:
    def __init__(self, actor, critic, memory):
        """
        DDPG for categorical action
        """
        dtype = torch.float64
        torch.set_default_dtype(dtype)
        self.device = torch.device('cuda:0')

        self.iter = 0

        self.actor = actor.to(self.device)
        self.target_actor = copy.deepcopy(actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), arglist.learning_rate)

        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), arglist.learning_rate)

        self.memory = memory
        self.nb_actions = 5

        self.target_actor.eval()
        self.target_critic.eval()

    def soft_update(self, target, source, tau):
        """
        Copies the parameters from source network (x) to target network (y) using the below update
        y = TAU*x + (1 - TAU)*y
        :param target: Target network (PyTorch)
        :param source: Source network (PyTorch)
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, target, source):
        """
        Copies the parameters from source network to target network
        :param target: Target network (PyTorch)
        :param source: Source network (PyTorch)
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def process_obs(self, obs):
        obs = np.array(obs, dtype='float32')
        return obs

    def process_action(self, actions):
        actions = np.argmax(actions, axis=-1)
        actions = actions.reshape(-1)
        return actions

    def process_reward(self, rewards):
        rewards = np.array(rewards, dtype='float32')
        return rewards

    def process_done(self, done):
        done = np.array(done, dtype='float32')
        return done

    def to_onehot(self, a1):
        a1 = to_categorical(a1, num_classes=self.nb_actions)
        a1 = a1.astype('float32')
        return a1

    def get_exploitation_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = torch.from_numpy(state).to(self.device)
        action = self.target_actor.forward(state).detach()
        action = action.data.numpy().to(self.device)

        return action

    def get_exploration_action(self, state, t_1, t_2, t_3, t_4):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        t_1.tic()
        state = np.expand_dims(state, axis=0)
        t_1.toc()
        t_2.tic()
        state = torch.from_numpy(state).to(self.device)
        t_2.toc()
        t_3.tic()
        action = self.actor.forward(state).detach()
        t_3.toc()
        t_4.tic()
        new_action = action.data.cpu().numpy()  # + (self.noise.sample() * self.action_lim)
        t_4.toc()
        return new_action

    def optimize(self, t1,t2,t3,t4,t5):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        s1, a1, r1, s2, d = self.memory.sample(arglist.batch_size)
        t1.tic()
        s1 = self.process_obs(s1)
        a1 = self.to_onehot(a1)
        r1 = self.process_reward(r1)
        s2 = self.process_obs(s2)
        d = self.process_done(d)
        t1.toc()

        t2.tic()
        s1 = torch.from_numpy(s1).to(self.device)
        a1 = torch.from_numpy(a1).to(self.device)
        r1 = torch.from_numpy(r1).to(self.device)
        s2 = torch.from_numpy(s2).to(self.device)
        d = torch.from_numpy(d).to(self.device)
        t2.toc()
        t3.tic()
        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a2 = self.target_actor.forward(s2).detach()
        q_next = torch.squeeze(self.target_critic.forward(s2, a2).detach())
        # y_exp = r + gamma*Q'( s2, pi'(s2))
        y_expected = r1 + GAMMA * torch.mul(q_next, torch.stack([(1. - d)]*3, dim=1))
        # y_pred = Q( s1, a1)
        y_predicted = torch.squeeze(self.critic.forward(s1, a1))
        # compute critic loss, and update the critic
        # loss_critic = F.smooth_l1_loss(y_predicted.cpu(), y_expected.cpu()) # todo: changed shrinked 0..2
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        t3.toc()

        t4.tic()
        # ---------------------- optimize actor ----------------------
        pred_a1 = self.actor.forward(s1)
        loss_actor = -1*torch.sum(self.critic.forward(s1, pred_a1))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # todo : if it's removed, it saves 0.6 seconds
        self.actor_optimizer.step()
        t4.toc()
        t5.tic()
        self.soft_update(self.target_actor, self.actor, arglist.tau)
        self.soft_update(self.target_critic, self.critic, arglist.tau)
        t5.toc()
        return loss_actor, loss_critic

    def save_models(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
        print('Models saved successfully')

    def load_models(self, episode):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
        print('Models loaded succesfully')

    def save_training_checkpoint(self, state, is_best, episode_count):
        """
        Saves the models, with all training parameters intact
        :param state:
        :param is_best:
        :param filename:
        :return:
        """
        filename = str(episode_count) + 'checkpoint.path.rar'
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')