from .abstract_policy import AbstractPolicy
import torch
import numpy as np
from src.env.environment import Environment


class NeuralNetPolicy(AbstractPolicy):
    """
    Abstract class of any policy

    This class is meant for programming purposes and can't be instantiated
    """

    def __init__(
            self,
            environment: Environment,
            hidden_size: int = 500,
            learning_rate: float = 0.003,
    ):
        super().__init__(environment)

        # We make a simple neural net model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.environment.n_observations, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.environment.n_actions),
            torch.nn.Softmax(dim=0)
        )

        # Optimizer of the model using backpropagation when we'll compute the loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def predict_proba(self, state: np.array) -> np.array:
        return self.model(torch.from_numpy(state).float()).data.numpy()

    def learn_one(self, transitions: list, gamma: float):
        # Transitions stores the history of (states, actions, rewards) at each episode
        # We want to fetch the rewards only here
        rewards = np.array([reward for (state, action, reward) in transitions])

        # We would like to compute the total return at each episode, so first we initialize it
        # We will approximate it by adding the rewards from some state
        # in the episode until the end of the episode using gamma
        G = np.zeros(len(transitions))
        for i in range(len(transitions)):
            # We compute the expected return of each episode
            G_i = 0
            power = 0
            for j in range(i, len(transitions)):
                G_i += (gamma ** power) * rewards[j]
                power += 1
            # Then we store it into the total return
            G[i] = G_i

        # We normalize the total return for numerical stability
        G = G/np.max(G)
        # We turn the arrays into tensors for computational usage in the network
        G = torch.FloatTensor(G)
        states = torch.FloatTensor([state for (state, action, reward) in transitions])
        actions = torch.FloatTensor([action for (state, action, reward) in transitions])
        # We make the action predictions for each state (probability of each action being taken at each state)
        predictions = self.model(states)
        # We now fetch the probability of the chosen action at each state
        proba_action_at_states = predictions.gather(dim=1, index=actions.long().view(-1, 1)).squeeze()

        # We compute the opposite of the gradient of the loss so it's in the form of the usual gradient descent
        # (though what we're trying to achieve is technically a gradient ascent)
        loss = -torch.sum(torch.log(proba_action_at_states) * G)

        # We do a backpropagation of the neural network to optimize the parameters with the updated loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
