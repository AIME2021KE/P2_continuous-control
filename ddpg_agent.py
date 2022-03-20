# KAE 3/19/2022: Most of this code was used as is from the DDPG code provided in the 
#  Udacity mini-project on the pendulum

# The sections on the multiple agents (didn't realize that was what we were doing; 
#  never saw the ability to do a single agent) were obtained from the following source
#  obtained by searched the internet for "how to handle mutliple agents in ddpg pytorch" 
#  and found the following link: 
# https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents
#   by Mike Richardson
# Inside that code we have the following header:
""""
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code Expanded and Adapted from Code provided by Udacity DRL Team, 2018.
"""
# which is provided here for completeness. Each code line / snippet that used information
#  from this source is documented below individually as the rest of the code comes
#  from the Udacity mini-project using DDPG on the pendulum

import numpy as np
import random
import copy
from collections import namedtuple, deque

from ddpg_model import Actor, Critic
#from MYreplay_buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

# commented out are pretty much as is from the DDPG example
#  the used terms were from the multi-agent help, which acheived 
#  exceeding the training score in just under 120 episodes
# https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
#BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
# NOTE: 3/19/2022 -- this value didn't go anywhere, so dropping to 1e-4
#LR_CRITIC = 1e-3        # learning rate of the critic
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# KAE 3/18/2022: This is a modified version of the original ddpg for a single agent,
#  but to allow multiple agents
# help was provided from https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents
# concerning how to get started on the multiple agents portion
#  the original example made modifications to the OUNoise class to reset with a diminising sigma, but we ignored that 
# we also try to maintain the nomenclature of the various parameters without an s as a 
#  singular value and with an s to be the set of agents of parameters


class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents employed
            random_seed (int): random seed
        """
        # save state, action and number of agent sizes along with our initial seed
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)

        # Actor Networks (w/ Target and local Networks)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target and local Networks)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, 
                                           weight_decay=WEIGHT_DECAY)

        # Noise process, with num_agents
# help was provided from https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents
#   define the size of the noise to NOW be a tuple of num_agents, action_size) 
#     instead of just action size
        self.noise = OUNoise((num_agents, action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    def step(self, states, actions, rewards, 
             next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # KAE 3/18/2022: for some strange reason get list indices must be integers or slices, not tuple python
        #  error message at this point AT THE COMMENTS???, appeared to be due to having the buffer class in here; but perhaps just not closing the environment, clearing restart,
        #  so removed it
        # KAE 3/18/2022: this is the key area where we have to read in all the agents together and then
        #  add them into our buffer separately
        # Save experience / reward for each agent
        for agent in range(self.num_agents):
# help was provided from https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents
#    add each tuple set (state, action, reward, next_state, done) to the memory buffer
            self.memory.add(states[agent,:], actions[agent,:], rewards[agent], 
                            next_states[agent,:], dones[agent])

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)


    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        # KAE 3/18/2022: looks like we now are bringing in number of agents worth of scores
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size))
        self.actor_local.eval()
        with torch.no_grad():
# help was provided from https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents
#    for each agent get an action from the local (actor) network given the individual states
            for agent in range(self.num_agents):
                actions[agent,:] = self.actor_local(states[agent,:]).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

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

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        # KAE 3/18/2022: the next line (commented) from the original DDPG code has a problem, self.size is a tuple 
        #   and presumably provides the right addition of terms....
#        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
# help was provided from https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents
#   use the 
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

# NOTE 3/19/2022: adding back the replay buffer as it was previously    
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

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        