"""DDPG agent for the Reacher continuous-control environment."""
import copy
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class OUNoise:
    def __init__(self, size, seed=27, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta, self.sigma = theta, sigma
        self.state = np.copy(self.mu)
        random.seed(seed)

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*self.mu.shape)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    Experience = namedtuple("E", ("state", "action", "reward", "next_state", "done"))
    def __init__(self, capacity=int(1e6), batch_size=128, seed=27):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        random.seed(seed)
    def add(self, *e): self.memory.append(self.Experience(*e))
    def sample(self):
        b = random.sample(self.memory, self.batch_size)
        s  = torch.from_numpy(np.vstack([e.state      for e in b])).float().to(DEVICE)
        a  = torch.from_numpy(np.vstack([e.action     for e in b])).float().to(DEVICE)
        r  = torch.from_numpy(np.vstack([e.reward     for e in b])).float().to(DEVICE)
        ns = torch.from_numpy(np.vstack([e.next_state for e in b])).float().to(DEVICE)
        d  = torch.from_numpy(np.vstack([e.done       for e in b]).astype(np.uint8)).float().to(DEVICE)
        return s, a, r, ns, d
    def __len__(self): return len(self.memory)


class DDPGAgent:
    def __init__(self, state_size=33, action_size=4, num_agents=20,
                 lr_actor=1e-4, lr_critic=1e-3, weight_decay=0,
                 buffer_size=int(1e6), batch_size=128,
                 gamma=0.99, tau=1e-3,
                 update_every=20, num_updates=10):
        self.num_agents = num_agents
        self.actor_local  = Actor(state_size, action_size).to(DEVICE)
        self.actor_target = Actor(state_size, action_size).to(DEVICE)
        self.actor_opt    = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        self.critic_local  = Critic(state_size, action_size).to(DEVICE)
        self.critic_target = Critic(state_size, action_size).to(DEVICE)
        self.critic_opt    = optim.Adam(self.critic_local.parameters(),
                                        lr=lr_critic, weight_decay=weight_decay)

        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.noise  = OUNoise((num_agents, action_size))
        self.gamma, self.tau = gamma, tau
        self.update_every, self.num_updates = update_every, num_updates
        self.t_step = 0

    def act(self, states, add_noise=True):
        states = torch.from_numpy(states).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions = actions + self.noise.sample()
        return np.clip(actions, -1.0, 1.0)

    def step(self, states, actions, rewards, next_states, dones):
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.memory.add(s, a, r, ns, d)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.memory.batch_size:
            for _ in range(self.num_updates):
                self._learn(self.memory.sample())

    def reset(self):
        self.noise.reset()

    def _learn(self, batch):
        s, a, r, ns, d = batch
        next_actions = self.actor_target(ns)
        with torch.no_grad():
            q_target_next = self.critic_target(ns, next_actions)
        q_target = r + self.gamma * q_target_next * (1 - d)
        q_expected = self.critic_local(s, a)
        critic_loss = F.mse_loss(q_expected, q_target)
        self.critic_opt.zero_grad(); critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_opt.step()

        pred_actions = self.actor_local(s)
        actor_loss = -self.critic_local(s, pred_actions).mean()
        self.actor_opt.zero_grad(); actor_loss.backward()
        self.actor_opt.step()

        self._soft(self.actor_local,  self.actor_target)
        self._soft(self.critic_local, self.critic_target)

    def _soft(self, src, dst):
        for s, t in zip(src.parameters(), dst.parameters()):
            t.data.copy_(self.tau * s.data + (1.0 - self.tau) * t.data)
