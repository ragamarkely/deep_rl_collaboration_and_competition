from config import BATCH_SIZE, BUFFER_SIZE, NUM_AGENTS, UPDATE_EVERY
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
        self.agents = [DDPG(state_size, action_size, i) for i in range(NUM_AGENTS)]
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.num_agents = NUM_AGENTS
        self.t_step = 0

    def reset(self):
        """
        Reset MADDPG.
        """
        self.t_step = 0
        for agent in self.agents:
            agent.noise.reset()

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
        return np.array([
            agent.act(state, add_noise) for agent, state in zip(self.agents, states)
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
        self.memory.add(states,actions,rewards,next_states,dones)

        if self.t_step % UPDATE_EVERY == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            for agent in self.agents:
                agent.learn(experiences)

        self.t_step += 1

    def save(self):
        """
        Save network parameters.
        """
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f"checkpoint_actor_local_{i}.pth")
            torch.save(agent.actor_target.state_dict(), f"checkpoint_actor_target_{i}.pth")
            torch.save(agent.critic_local.state_dict(), f"checkpoint_critic_local_{i}.pth")
            torch.save(agent.critic_target.state_dict(), f"checkpoint_critic_target_{i}.pth")