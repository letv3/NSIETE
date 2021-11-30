import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class Net(nn.Module):
    # def __init__(self, gamma, img_stack, alpha=1e-3):
    #     super(Net, self).__init__()
    #     self.img_stack = img_stack
    #     self.gamma = gamma
    #
    #     self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
    #         nn.Conv2d(self.img_stack, 8, kernel_size=4, stride=2),
    #         nn.ReLU(),  # activation
    #         nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
    #         nn.ReLU(),  # activation
    #         nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
    #         nn.ReLU(),  # activation
    #         nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
    #         nn.ReLU(),  # activation
    #         nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
    #         nn.ReLU(),  # activation
    #         nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
    #         nn.ReLU(),  # activation
    #     )  # output shape (256, 1, 1)
    #     self.v = nn.Sequential(nn.Linear(256, 100),
    #                            nn.ReLU(), nn.Linear(100, 1))
    #     self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
    #     self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
    #     self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
    #     self.apply(self._weights_init)
    #     self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def __init__(self, observation_space, gamma, alpha=1e-3):
        super(Net, self).__init__()
        self.input_dim = observation_space
        self.gamma = gamma

        self.simplenet = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.stateavier_uniform_(
                m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, state):
        state = self.cnn_base(state)
        state = state.view(-1, 256)
        v = self.v(state)
        state = self.fc(state)
        alpha = self.alpha_head(state) + 1
        beta = self.beta_head(state) + 1

        return (alpha, beta), v


class Actor(nn.Module):
    def __init__(self, obs_space, action_space, l1_size=400, l2_size=300):
        super(Actor, self).__init__()
        self.action_space = action_space

        self.layer1 = nn.Linear(obs_space, l1_size)
        self.layer2 = nn.Linear(l1_size, l2_size)
        self.layer3 = nn.Linear(l2_size, action_space)

    def forward(self, state):
        state = F.relu(self.layer1(state))
        state = F.relu(self.layer2(state))
        state = self.layer3(state)

        # state = T.tanh(state) * T.from_numpy(self.action_space.high).float()
        return state


class Critic(nn.Module):
    def __init__(self, obs_space, action_space, l1_size=400, l2_size=300):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_space, l1_size)
        self.fc2 = nn.Linear(l1_size + action_space, l2_size)
        self.fc3 = nn.Linear(l2_size, 1)

    def forward(self, state, action):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(T.cat([state, action], dim=1)))
        state = self.fc3(state)

        return state
