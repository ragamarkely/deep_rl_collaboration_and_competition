from collections import deque
import random
from typing import NamedTuple

from config import DEVICE
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

        states = torch.from_numpy(np.vstack([e.states for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.actions for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        Get the number of experience tuples in the memory.
        """
        return len(self.memory)
