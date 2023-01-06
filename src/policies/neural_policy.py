from .abstract_policy import AbstractPolicy
import torch
import numpy as np
from src.env.environment import Environment


class NeuralNetPolicy(AbstractPolicy):
    """
    Neural network modelisation of a policy

    Parameter
    ---------
    environment
        Environment in which the policy should be training
    hidden_layer_size
        Number of neurons in the hidden layer of the neural network
    learning_rate
        Learning rate of the Adam optimizer
    """

    def __init__(
            self,
            environment: Environment,
            hidden_layer_size: int = 500,
            learning_rate: float = 0.003,
    ):
        super().__init__(environment)

        # We make a simple neural net model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.environment.n_observations, hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_size, self.environment.n_actions),
            torch.nn.Softmax(dim=0)
        )

        # Optimizer of the model using backpropagation when we'll compute the loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def predict_proba(self, state: torch.Tensor) -> torch.Tensor:
        return self.model(torch.from_numpy(state).float())

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
