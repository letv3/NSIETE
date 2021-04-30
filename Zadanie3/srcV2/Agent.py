from Nets import Actor, Critic
import torch
import os
import numpy as np
import random
from Memory import Memory


class Agent(object):
    def __init__(self,
                 actor_lr,
                 critic_lr,
                 gamma,
                 batch_size,
                 min_replay_size,
                 memory_capacity,
                 tau,
                 n_inputs,
                 n_actions,
                 **kwargs):

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Neural networks
        # Policy Network
        self.actor = Actor(alpha=actor_lr, n_inputs=n_inputs, n_actions=n_actions).to(self.device)
        self.target_actor = Actor(alpha=actor_lr, n_inputs=n_inputs, n_actions=n_actions).to(self.device)
        self.actor_optimizer = self.actor.optimizer

        # Evaluation Network
        self.critic = Critic(beta=critic_lr, n_inputs=n_inputs, n_actions=n_actions).to(self.device)
        self.target_critic = Critic(beta=critic_lr, n_inputs=n_inputs, n_actions=n_actions).to(self.device)
        self.critic_optimizer = self.critic.optimizer

        # Sync weights
        self.weights_sync()

        # Replay buffer
        self.min_replay_size = min_replay_size
        self.memory = Memory(memory_capacity)

        # Constants
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_actions = n_actions

    def action(self, observation):
        states = torch.from_numpy(observation).type(torch.float).to(self.device)
        states = states.view((-1, *states.shape))
        return self.actor(states)[0]

    def target_action(self, observation):
        states = torch.from_numpy(observation).type(torch.float).to(self.device)
        states = states.view((-1, *states.shape))
        return self.target_actor(states)[0]

    def random_action(self):
        return torch.FloatTensor(self.n_actions).uniform_(-1, 1).to(self.device)

    def train(self):

        # Get samples from memory
        states, actions, rewards, new_states, done =self.memory.sample(self.batch_size)

        # Convert samples to tensors
        states = torch.tensor(states, dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        new_states = torch.from_numpy(np.stack(new_states)).to(dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.float, device=self.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        # Loss statistics
        loss_results = {}

        # Train critic network
        with torch.no_grad():
            targets = rewards + self.gamma * (1 - done) * self.target_critic(new_states, self.target_actor(new_states))
        predicted = self.critic(states, actions)
        loss = ((targets - predicted) ** 2).mean()
        loss_results['critic_loss'] = loss.data

        self.critic_optimizer.zero_grad()
        self.critic.train()
        loss.backward()
        self.critic_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = False

        states = self.memory.sample(self.batch_size)[0]

        # Convert samples to tensors
        states = torch.tensor(states, dtype=torch.float, device=self.device)

        self.critic.eval()
        self.actor.eval()

        # Train actor network
        predicted = self.critic(states, self.actor(states))
        loss = -predicted.mean()
        loss_results['actor_loss'] = loss.data
        self.actor_optimizer.zero_grad()
        self.actor.train()
        loss.backward()
        self.actor_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        return loss_results

    def store_experience(self, transition):
        self.memory.store(transition)

    def update(self):
        with torch.no_grad():
            for actor_param, target_actor_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_actor_param.data = (1.0 - self.tau) * target_actor_param.data + self.tau * actor_param.data

            for critic_param, target_critic_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_critic_param.data = (1.0 - self.tau) * target_critic_param.data + self.tau * critic_param.data

    def weights_sync(self):
        with torch.no_grad():
            for actor_param, target_actor_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_actor_param.data = actor_param.data

            for critic_param, target_critic_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_critic_param.data = critic_param.data

    def save_agent(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        target_actor_path = os.path.join(save_path, "target_actor_network.pth")
        torch.save(self.target_actor.state_dict(), target_actor_path)

        target_critic_path = os.path.join(save_path, "target_critic_network.pth")
        torch.save(self.target_critic.state_dict(), target_critic_path)

        actor_path = os.path.join(save_path, "actor_network.pth")
        torch.save(self.actor.state_dict(), actor_path)

        critic_path = os.path.join(save_path, "critic_network.pth")
        torch.save(self.critic.state_dict(), critic_path)

    def load_agent(self, save_path):
        actor_path = os.path.join(save_path, "actor_network.pth")
        self.actor.load_state_dict(torch.load(actor_path))
        self.actor.eval()

        target_actor_path = os.path.join(save_path, "target_actor_network.pth")
        self.target_actor.load_state_dict(torch.load(target_actor_path))
        self.target_actor.eval()

        critic_path = os.path.join(save_path, "critic_network.pth")
        self.critic.load_state_dict(torch.load(critic_path))
        self.critic.eval()

        target_critic_path = os.path.join(save_path, "target_critic_network.pth")
        self.target_critic.load_state_dict(torch.load(target_critic_path))
        self.target_critic.eval()

        self.weights_sync()

    def __str__(self):
        return str(self.actor) + str(self.critic)
