from collections import deque
import random
from typing import NamedTuple

from config import DEVICE, NUM_AGENTS
import numpy as np 
import torch

class Experience(NamedTuple):
    """
    Experience tuple.
    """
    states: np.ndarray 
    actions: int
    rewards: float 
    next_states: np.ndarray 
    dones: bool


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        """
        Initialize replay buffer.

        Params
        ======
        buffer_size
        batch_size
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, states, actions, rewards, next_states, dones):
        """
        Add experience tuple into memory.
        
        Params
        ======
        states
        actions
        rewards
        next_states
        dones
        """
        self.memory.append(Experience(states, actions, rewards, next_states, dones))

    def sample(self):
        """
        Randomly sample a batch of experiences from memory.

        Returns
        =======
        tuple of (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        states = [
            torch.from_numpy(
                np.vstack([e.states[i] for e in experiences if e is not None])
            ).float().to(DEVICE) 
            for i in range(NUM_AGENTS)
        ]
        actions = [
            torch.from_numpy(
                np.vstack([e.actions[i] for e in experiences if e is not None])
            ).float().to(DEVICE) 
            for i in range(NUM_AGENTS)
        ]
        rewards = [
            torch.from_numpy(
                np.vstack([e.rewards[i] for e in experiences if e is not None])
            ).float().to(DEVICE) 
            for i in range(NUM_AGENTS)
        ]
        next_states = [
            torch.from_numpy(
                np.vstack([e.next_states[i] for e in experiences if e is not None])
            ).float().to(DEVICE) 
            for i in range(NUM_AGENTS)
        ]
        dones = [
            torch.from_numpy(
                np.vstack([e.dones[i] for e in experiences if e is not None]).astype(np.uint8)
            ).float().to(DEVICE) 
            for i in range(NUM_AGENTS)
        ]
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        Get the number of experience tuples in the memory.
        """
        return len(self.memory)
