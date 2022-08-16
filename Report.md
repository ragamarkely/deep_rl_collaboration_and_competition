# Project 3: Competition and Collaboration

## Implementation
In this project, [Multi-Agent Deep Deterministic Policy Gradient (MADDPG)](https://arxiv.org/abs/1706.02275) algorithm is used to solve the Tennis environment.

Each of the Actor's and Critic's local and target networks is a Deep Neural Network with 2 hidden layers of fully connected networks. The first hidden layer has 256 units and the second hidden layer has 128 units. In addition, batch normalization was used in both the Actor and Critic networks. In this implementation, the networks are updated 10 times after every 10 time steps. 

## Future Work
The performance (score improvement per episode and stability) of the model is extremely sensitive to the hyperparameter values. For the future, we need to study the impact of these hyperparameters on the training performance in order to strategize the hyperparameter search. Some examples from this project are highlighted below:
1. The number of units in the hidden layer. For example, changing the number of hidden units in the second hidden layer from 256 to 128 caused the training score to remain low (< 0.001) most of the time without any sign of improvement.
2. The max number of time steps for each episode. It has significant impact on model performance and the magnitude of the effect is dependent on other hyperparameter values.
3. Batch and buffer sizes. Increasing the sizes help the performance, but it is unclear what should be the optimum size and how do the sizes interact with other hyperparameters.
