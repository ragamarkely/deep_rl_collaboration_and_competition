import torch

# Learning parameters
LR_ACTOR = 1e-4         # Actor learning rate
LR_CRITIC = 1e-3        # Critic learning rate
TAU = 1e-3              # Soft update parameter
GAMMA = 0.99            # Reward discount factor
WEIGHT_DECAY = 0.       # Weight decay for Critic
UPDATE_EVERY = 5        # Network update frequency
UPDATE_COUNT = 3       # Number of learning steps per update

# Parameters for memory buffer
BUFFER_SIZE = int(1e6)  # Memory replay buffer size 
BATCH_SIZE = 1024        # Batch size for training

# Noise parameters
MU = 0.
THETA = 0.15
SIGMA = 0.2

# Other parameters
NUM_AGENTS = 2
RANDOM_SEED = 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

