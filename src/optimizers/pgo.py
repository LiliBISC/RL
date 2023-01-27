import sys
sys.path.append('C:/Users/lilia/OneDrive/Documents/GitHub/RL/src/policies')
sys.path.append('C:/Users/lilia/OneDrive/Documents/GitHub/RL/src/optimizers')
from abstract_optimizer import AbstractOptimizer
import numpy as np
from abstract_policy import AbstractPolicy
import torch


class PGO(AbstractOptimizer):
    """
    Implementation of Vanilla Policy Gradient Optimizer

    Parameter
    ---------
    policy
        Policy to optimize (living in a given environment)
    horizon
        Horizon of the discounted setting
    gamma
        Discounted setting parameter
    """

    def __init__(
            self,
            policy: AbstractPolicy,
            horizon: int,
            gamma: float
    ):
        super().__init__(policy, horizon, gamma)

    def step(self) -> int:
        # Let the agent play in the environment with the given horizon
        score, transitions = self.policy.play(self.horizon)

        # Now we improve the policy with the transitions experience,
        # with the discounted setting parameter of the problem
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
                G_i += (self.gamma ** power) * rewards[j]
                power += 1
            # Then we store it into the total return
            G[i] = G_i

        # We normalize the total return for numerical stability
        G = G / np.max(G)
        # We turn the arrays into tensors for computational usage in the network
        G = torch.FloatTensor(G)
        states = np.array([state for (state, action, reward) in transitions])
        actions = torch.FloatTensor([action for (state, action, reward) in transitions])
        # We make the action predictions for each state (probability of each action being taken at each state)
        predictions = self.policy.predict_proba(states)
        # We now fetch the probability of the chosen action at each state
        proba_action_at_states = predictions.gather(dim=1, index=actions.long().view(-1, 1)).squeeze()

        # We compute the opposite of the gradient of the loss so it's in the form of the usual gradient descent
        # (though what we're trying to achieve is technically a gradient ascent)
        loss = -torch.sum(torch.log(proba_action_at_states) * G)

        # We do a backpropagation of the neural network to optimize the parameters with the updated loss
        self.policy.optimize(loss)

        return score
