from config import (
    BATCH_SIZE, 
    BUFFER_SIZE, 
    NUM_AGENTS, 
)
from ddpg import DDPG
from memory import ReplayBuffer
import numpy as np
import torch

class MADDPG:
    """
    Manage DDPG agents in training and interacting with environment.
    """
    def __init__(self, state_size, action_size):
        """
        Initialize MADDPG.
        """
        self.state_size = state_size
        self.action_size = action_size 
        self.agents = [DDPG(state_size, action_size) for _ in range(NUM_AGENTS)]
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.num_agents = NUM_AGENTS
        self.t_step = 0

    def act(self, states, add_noise=True):
        """
        Get actions for each DDPG agent given the state.

        Params
        ======
        states
        add_noise

        Returns
        =======
        actions
        """
        return np.concatenate([
            agent.act(state, add_noise).reshape(1, 2) 
            for agent, state in zip(self.agents, states)
        ])

    def step(self, states, actions, rewards, next_states, dones):
        """
        Save experience in memory and learn.

        Params
        ======
        states
        actions
        rewards
        next_states
        dones
        """
        for i in range(NUM_AGENTS):
            self.agents[i].step(states[i], actions[i], rewards[i], next_states[i], dones[i])
            # Save experience for the other agents.
            self.agents[1 - i].memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def save(self):
        """
        Save network parameters.
        """
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f"checkpoint_actor_local_{i}.pth")
            torch.save(agent.actor_target.state_dict(), f"checkpoint_actor_target_{i}.pth")
            torch.save(agent.critic_local.state_dict(), f"checkpoint_critic_local_{i}.pth")
            torch.save(agent.critic_target.state_dict(), f"checkpoint_critic_target_{i}.pth")