import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from actor_critic_networks import Actor_2HL_BN, Critic_2HL_BN, Actor_1HL, Critic_3HL_leaky

LR_ACTOR     = 0.0001    # learning rate of the actor 
LR_CRITIC    = 0.001     # learning rate of the critic
WEIGHT_DECAY = 0.0       # L2 weight decay

class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, actor_net, critic_net):
        """Initialize an DDPGAgent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            actor_net (torch.nn.Module): neural net for the actor 
            critic_net (torch.nn.Module): neural net for the critic 
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = actor_net(state_size, action_size, random_seed).to('cpu')
        self.actor_target = actor_net(state_size, action_size, random_seed).to('cpu')
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.hard_update(self.actor_local, self.actor_target)
            
        # Critic Network (w/ Target Network)
        # Remember that in MADDPG, actors are decentralized, whereas critics are centralized.
        self.critic_local = critic_net(state_size*2, action_size*2, random_seed).to('cpu')
        self.critic_target = critic_net(state_size*2, action_size*2, random_seed).to('cpu')
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.hard_update(self.critic_local, self.critic_target)
            
        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to('cpu')
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
        
    def reset(self):
        self.noise.reset()
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def hard_update(self, local_model, target_model):
        """Hard update model parameters.

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)            

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, seed, mu=0.0, theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.shape = shape
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.shape)
        self.state = x + dx
        return self.state
