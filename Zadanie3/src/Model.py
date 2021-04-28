import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from src.net import Net

from src.net import Actor, Critic
from src.Memory import Memory

from gym import spaces


class Agent:
    max_grad_norm = 0.5
    clip_param = 0.1
    ppo_epoch = 10
    buffer_capacity, batch_size = 5000, 1024

    def __init__(self,env,alpha=1e-4, gamma=0.99, tau=1e-2):

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.training_step = 0
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        # Actor-Critic initialization
        self.actor = Actor(
            obs_space=self.observation_space.shape[0],
            action_space=self.action_space.n if hasattr(self.action_space, "n") else self.action_space.shape[0]
        ).double().to(self.device)

        self.actor_target = Actor(
            obs_space=self.observation_space.shape[0],
            action_space=self.action_space.n if hasattr(self.action_space, "n") else self.action_space.shape[0]
        ).double().to(self.device)

        self.critic = Critic(
            obs_space=self.observation_space.shape[0],
            action_space=self.action_space.n if hasattr(self.action_space, "n") else self.action_space.shape[0]
        ).double().to(self.device)

        self.critic_target = Critic(
            obs_space=self.observation_space.shape[0],
            action_space=self.action_space.n if hasattr(self.action_space, "n") else self.action_space.shape[0]
        ).double().to(self.device)

        # Duplicate networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Transition memory initialization
        self.transition = np.dtype(
            [('s', np.float64, self.observation_space.shape), ('a', np.float64, self.action_space.shape),
             ('r', np.float64, (1,)), ('s_n', np.float64, self.observation_space.shape), ("d", np.float64, (1,))])
        self.memory = Memory(self.buffer_capacity, self.transition)

        # Training
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha)

    def select_action(self, state):
        state_x = T.from_numpy(state).to(self.device)
        action = self.actor.forward(state_x)

        if type(self.action_space) == spaces.Box:
            # action = T.tanh(action).detach().numpy()
            action = T.tanh(action).cpu().detach().numpy()
        elif type(self.action_space) == spaces.Discrete:
            action = action  # TODO Implement action handling if disccrete
        else:
            raise Exception("Invalid action space type.")
        return action
        # return action.clip(min=self.action_space.low, max=self.action_space.high)  # TODO: clip action?

    def update(self):  # TODO Implement update
        states, actions, rewards, next_states, _ = self.memory.sample(self.batch_size)
        states = T.DoubleTensor(states).to(self.device)
        actions = T.DoubleTensor(actions).to(self.device)
        rewards = T.DoubleTensor(rewards).to(self.device)
        next_states = T.DoubleTensor(next_states).to(self.device)

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))






class Agent_old:
    max_grad_norm = 0.5
    clip_param = 0.1
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

        self.transition = np.dtype(
            [('s', np.float64, (self.img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
             ('r', np.float64), ('s_n', np.float64, (self.img_stack, 96, 96))])
        self.training_step = 0
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        self.net = Net(alpha=self.alpha, gamma=self.gamma,
                       img_stack=self.img_stack).double().to(self.device)

        self.buffer = np.empty(self.buffer_capacity, dtype=self.transition)
        self.counter = 0

    def select_action(self, state):
        state = T.from_numpy(state).double().to(self.device).unsqueeze(0)
        with T.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)
        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()

        return action, a_logp

    def save_param(self, name):
        T.save(self.net.state_dict(), 'data/model/' + name + '.pkl')

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def load_param(self, name):
        self.net.load_state_dict(T.load('data/model/' + name + '.pkl'))

    def update(self):
        self.training_step += 1

        s = T.tensor(self.buffer['s'], dtype=T.double).to(self.device)
        a = T.tensor(self.buffer['a'], dtype=T.double).to(self.device)
        r = T.tensor(self.buffer['r'], dtype=T.double).to(
            self.device).view(-1, 1)
        s_n = T.tensor(self.buffer['s_n'], dtype=T.double).to(self.device)

        old_a_logp = T.tensor(self.buffer['a_logp'], dtype=T.double).to(
            self.device).view(-1, 1)

        with T.no_grad():
            target_v = r + self.gamma * self.net(s_n)[1]
            adv = target_v - self.net(s)[1]

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = T.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = T.clamp(ratio, 1.0 - self.clip_param,
                                1.0 + self.clip_param) * adv[index]
                action_loss = -T.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(
                    self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.net.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.net.optimizer.step()
