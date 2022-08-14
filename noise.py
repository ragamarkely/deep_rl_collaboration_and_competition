from config import MU, THETA, SIGMA
import numpy as np

class OUNoise:
    """
    Ornstein-Uhlenbeck process to add randomness (exploration) to action selection.
    """
    def __init__(self, size, mu=MU, theta=THETA, sigma=SIGMA):
        """
        Initialize noise parameters.
        """
        self.size = size 
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset() 

    def reset(self):
        """
        Reset noise.
        """
        self.state = np.copy(self.mu)

    def sample(self):
        """
        Get noise sample.
        """
        x = self.state 
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx 
        return self.state