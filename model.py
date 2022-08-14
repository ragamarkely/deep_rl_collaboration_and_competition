from config import NUM_AGENTS, RANDOM_SEED
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    """
    Initialize hidden layers.
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """
    Actor model.
    """
    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=256):
        """
        Initialize Actor model.
        
        Params
        ======
        state_size: state dimension
        action_size: action dimension
        fc1_units: first hidden layer dimension
        fc2_units: second hidden layer dimension
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(RANDOM_SEED)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weight parameters.
        """ 
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Get action given input state.

        Params
        ======
        state

        Returns
        =======
        actions
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """
    Critic model.
    """
    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=256):
        """
        Initialize Critic model.

        Params
        ======
        state_size: state dimension
        action_size: action dimension
        fc1_units: first hidden layer dimension
        fc2_units: second hidden layer dimension
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(RANDOM_SEED)
        self.fc1 = nn.Linear((state_size + action_size) * NUM_AGENTS, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.bn = nn.BatchNorm1d(fc1_units)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weight parameters.
        """ 
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Get Q value based on state and action.

        Params
        ======
        state
        action

        Returns
        =======
        Q-value
        """
        xs = torch.cat((state, action.float()), dim=1)
        x = F.leaky_relu(self.fc1(xs))
        x = self.bn(x)
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)
