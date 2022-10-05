import numpy as np
import random
import torch
from collections import namedtuple, deque
import torch.nn.functional as F
from deep_det_policy_grad_agent import DDPGAgent

BUFFER_SIZE      = 100000  # replay buffer size
BATCH_SIZE       = 256     # minibatch size
GAMMA            = 0.999   # discount factor
TAU              = 0.001   # for soft update of target parameters
UPDATE_EACH_STEP = 15      # episodes between learning process
NO_OF_UPDATES    = 5       # how many updates in a row
USE_CLIPPING     = True    # use clipping or not

class MADDPG():
    """Coordinates the agents of class 'DDPGAgent'"""
    def __init__(self, state_size, action_size, random_seed, actor_net, critic_net):
        """Initialize an MADDPG object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            actor_net (torch.nn.Module): neural net for each actor --> the same type for each actor
            critic_net (torch.nn.Module): neural net for each critic --> the same type for each critic 
        """
        super(MADDPG, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        # we need two agents, tennis racket left and tennis racket right 
        self.maddpg_agents = [DDPGAgent(state_size, action_size, random_seed, actor_net, critic_net), 
                              DDPGAgent(state_size, action_size, random_seed, actor_net, critic_net)
                             ]

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, states, actions, rewards, next_states, dones, step):
        """ Save experience in replay memory, and use random sample from buffer to learn """
 
        # Save experience / reward 
        self.memory.add(np.array(states).reshape(1,-1).squeeze(), 
                        np.array(actions).reshape(1,-1).squeeze(), 
                        rewards,
                        np.array(next_states).reshape(1,-1).squeeze(),
                        dones)
         
        # If enough samples in the replay memory and if it is time to update
        if (len(self.memory) > BATCH_SIZE) and (step % UPDATE_EACH_STEP == 0) :
             for _ in range(NO_OF_UPDATES):
                # tennis racket left
                experiences = self.memory.sample()   
                self.learn(experiences, GAMMA, tennis_racket_idx=0)
                # tennis racket right
                experiences = self.memory.sample()   
                self.learn(experiences, GAMMA, tennis_racket_idx=1)
                
    def act(self, states, noise):
        """ Simply apply 'act' function of DDPGAgent-class"""
        return [agent.act(state, noise) for agent, state in zip(self.maddpg_agents, states)]
                
    def reset(self):
        for ddpg_agent in self.maddpg_agents:
            ddpg_agent.reset()                
    
    def learn(self, experiences, gamma, tennis_racket_idx):
        """
        Update the policy. Remember that in MADDPG, actors are decentralized, 
        whereas critics are centralized.
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(states) -> action
            critic_target(all_states, all_actions) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            tennis_racket_idx (int): left (--> 0) or right (--> 1) racket
        """
        
        states, actions, rewards, next_states, dones = experiences
        
        # The following separation into actions and states with respect to agents 0 and 1 is necessary
        # due to the fact that actors are decentralized, whereas critics are centralized

        relevant_states      = states.index_select(1, torch.tensor([idx for idx in range(tennis_racket_idx * self.state_size, tennis_racket_idx * self.state_size + self.state_size)]).to('cpu'))
        relevant_actions     = actions.index_select(1, torch.tensor([idx for idx in range(tennis_racket_idx * self.action_size, tennis_racket_idx * self.action_size + self.action_size)]).to('cpu'))
        relevant_next_states = next_states.index_select(1, torch.tensor([idx for idx in range(tennis_racket_idx * self.state_size, tennis_racket_idx * self.state_size + self.state_size)]).to('cpu'))

        irrelevant_states      = states.index_select(1, torch.tensor([idx for idx in range((1-tennis_racket_idx) * self.state_size, (1-tennis_racket_idx) * self.state_size + self.state_size)]).to('cpu'))
        irrelevant_actions     = actions.index_select(1, torch.tensor([idx for idx in range((1-tennis_racket_idx) * self.action_size, (1-tennis_racket_idx) * self.action_size + self.action_size)]).to('cpu'))
        irrelevant_next_states = next_states.index_select(1, torch.tensor([idx for idx in range((1-tennis_racket_idx) * self.state_size, (1-tennis_racket_idx) * self.state_size + self.state_size)]).to('cpu'))
       
        all_states      = torch.cat((relevant_states, irrelevant_states), dim=1).to('cpu')
        all_actions     = torch.cat((relevant_actions, irrelevant_actions), dim=1).to('cpu')
        all_next_states = torch.cat((relevant_next_states, irrelevant_next_states), dim=1).to('cpu')
   
        current_agent = self.maddpg_agents[tennis_racket_idx]
        
        # ---------------------------- update critic ---------------------------- #
        all_next_actions = torch.cat((current_agent.actor_target(relevant_states), current_agent.actor_target(irrelevant_states)), dim=1).to('cpu') 
        Q_targets_next = current_agent.critic_target(all_next_states, all_next_actions)
                
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = current_agent.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        current_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        if USE_CLIPPING:
            torch.nn.utils.clip_grad_norm_(current_agent.critic_local.parameters(), 1)
        current_agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        all_actions_pred = torch.cat((current_agent.actor_local(relevant_states), current_agent.actor_local(irrelevant_states).detach()), dim = 1).to('cpu')      
        actor_loss = -current_agent.critic_local(all_states, all_actions_pred).mean()
        
        # Minimize the loss
        current_agent.actor_optimizer.zero_grad()
        actor_loss.backward()        
        current_agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        current_agent.soft_update(current_agent.critic_local, current_agent.critic_target, TAU)
        current_agent.soft_update(current_agent.actor_local, current_agent.actor_target, TAU) 
        
    def save_trained_agents(self, pth_filename_act, pth_filename_cri):
        """Save the trained agents
                Params
        ======
            pth_filename_act (str): name for the '*.pth'-file for actors
            pth_filename_cri (str): name for the '*.pth'-file for critics
        """
        for tennis_racket_idx, ddpg_agent in enumerate(self.maddpg_agents):
            actor_local_filename = pth_filename_act + '_' + str(tennis_racket_idx) + '.pth'
            critic_local_filename = pth_filename_cri + '_' + str(tennis_racket_idx) + '.pth'           
            torch.save(ddpg_agent.actor_local.state_dict(), actor_local_filename) 
            torch.save(ddpg_agent.critic_local.state_dict(), critic_local_filename)             
                          
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to('cpu')
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to('cpu')
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to('cpu')
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to('cpu')
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to('cpu')

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)            