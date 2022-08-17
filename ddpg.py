from config import (
    DEVICE, 
    GAMMA, 
    LR_ACTOR, 
    LR_CRITIC, 
    TAU,
    UPDATE_COUNT,
    UPDATE_EVERY, 
    WEIGHT_DECAY,
    BATCH_SIZE, 
    BUFFER_SIZE, 
)
from memory import ReplayBuffer
from model import Actor, Critic
from noise import OUNoise
import numpy as np
import torch
import torch.nn.functional as F

class DDPG:
    """
    DDPG agent.
    """
    def __init__(self, state_size, action_size):
        """
        Initialize DDPG agent object.

        Params
        ======
        state
        size
        action_size
        agent_index
        """
        self.action_size = action_size 
        self.state_size = state_size

        # Actor Network & Optimizer
        self.actor_local = Actor(state_size, action_size).to(DEVICE)
        self.actor_target = Actor(state_size, action_size).to(DEVICE)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network & Optimizer
        self.critic_local = Critic(state_size, action_size).to(DEVICE)
        self.critic_target = Critic(state_size, action_size).to(DEVICE)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Use the same weight for target and local networks.
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

        # Noise 
        self.noise = OUNoise(action_size)

        # Replay memory buffer
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

        # Time step tracker
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Save experience in memory and learn.

        Params
        ======
        state
        action
        reward
        next_state
        done
        """
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1

        if self.t_step % UPDATE_EVERY == 0 and len(self.memory) > BATCH_SIZE:
            for _ in range(UPDATE_COUNT):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """
        Get action given the state.

        Params
        ======
        state
        add_noise

        Returns
        =======
        action
        """
        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma):
        """
        Update network parameters from a batch of experience tuples.

        Params
        ======
        experiences: experience tuple
        gamma: discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ----------------------- Update critic -------------------------------#
        # Get predicted next state actions and Q values from target models.
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states.
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss.
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize loss.
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------- Update actor --------------------------------#       
        # Compute actor loss.
        actions_pred = self.actor_local(states)
        actor_loss = - self.critic_local(states, actions_pred).mean()
        # Minimize loss.
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------- Update target networks --------------------------#
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update target model parameters.

        Params
        ======
        local_model
        target_model
        TAU
        """
        for target_params, local_params in zip(target_model.parameters(), local_model.parameters()):
            target_params.data.copy_(tau * local_params.data + (1.0 - tau) * target_params.data)

    def hard_update(self, target_model, local_model):
        """
        Copy parameters from local model to target model.

        Params
        ======
        target_model
        local_model
        """
        for target_params, local_params in zip(target_model.parameters(), local_model.parameters()):
            target_params.data.copy_(local_params.data)
