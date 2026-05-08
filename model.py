"""Actor + Critic networks for DDPG on the Reacher continuous-control task."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / (fan_in ** 0.5)
    return -lim, lim


class Actor(nn.Module):
    """Maps a 33-dim state to a 4-dim continuous action in [-1, 1]."""
    def __init__(self, state_size=33, action_size=4, fc1=256, fc2=128, seed=27):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1)
        self.bn1 = nn.BatchNorm1d(fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*_hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*_hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Q-value head — concat (state-emb, action) before the second hidden layer."""
    def __init__(self, state_size=33, action_size=4, fc1=256, fc2=128, seed=27):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1)
        self.bn1 = nn.BatchNorm1d(fc1)
        self.fc2 = nn.Linear(fc1 + action_size, fc2)
        self.fc3 = nn.Linear(fc2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*_hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*_hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = F.relu(self.bn1(self.fc1(state)))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
