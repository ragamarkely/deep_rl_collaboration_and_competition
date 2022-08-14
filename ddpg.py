from config import DEVICE, GAMMA, LR_ACTOR, LR_CRITIC, NUM_AGENTS, TAU, WEIGHT_DECAY
from model import Actor, Critic
from noise import OUNoise
import numpy as np
import torch
import torch.nn.functional as F

class DDPG:
    """
    DDPG agent.
    """
    def __init__(self, state_size, action_size, agent_index):
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
        self.agent_index = agent_index

        self.actor_local = Actor(state_size, action_size).to(DEVICE)
        self.actor_target = Actor(state_size, action_size).to(DEVICE)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_local = Critic(state_size, action_size).to(DEVICE)
        self.critic_target = Critic(state_size, action_size).to(DEVICE)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

        self.noise = OUNoise(action_size)

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

    def learn(self, experiences):
        """
        Update network parameters from a batch of experience tuples.

        Params
        ======
        experiences: experience tuple
        """
        states, actions, rewards, next_states, dones = experiences
        all_states = torch.cat(states, dim=1).to(DEVICE)
        all_next_states = torch.cat(next_states, dim=1).to(DEVICE)
        all_actions = torch.cat(actions, dim=1).to(DEVICE)

        next_actions = [actions[i].clone() for i in range(NUM_AGENTS)]
        next_actions[self.agent_index] = self.actor_target(next_states[self.agent_index])
        all_next_actions = torch.cat(next_actions, dim=1).to(DEVICE)

        Q_targets_next = self.critic_target(all_next_states, all_next_actions)
        Q_target = rewards[self.agent_index] + GAMMA * Q_targets_next * (1 - dones[self.agent_index])
        Q_expected = self.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        actions_pred = [actions[i].clone() for i in range(NUM_AGENTS)]
        actions_pred[self.agent_index] = self.actor_local(states[self.agent_index])
        all_actions_pred = torch.cat(actions_pred, dim=1).to(DEVICE)

        self.actor_optimizer.zero_grad()
        actor_loss = - self.critic_local(all_states, all_actions_pred).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

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
