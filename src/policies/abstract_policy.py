from abc import ABC, abstractmethod
import sys
sys.path.append('C:/Users/lilia/OneDrive/Documents/GitHub/RL/src/env')
from environment import Environment
import numpy as np
import torch
from torch.distributions.categorical import Categorical


class AbstractPolicy(ABC, object):
    """
    Abstract class of any policy

    This class is meant for programming purposes and can't be instantiated

    Parameter
    ---------
    environment
        Environment in which the policy should be training
    """

    def __init__(
            self,
            environment: Environment
    ):
        self.environment = environment
        # There's only 1 environment to go into in that situation
        self.actor = self.layer_init(torch.nn.Linear(2, 1), std=0.01)
        self.critic = self.layer_init(torch.nn.Linear(2, 1), std=1)
        # Initialize the model in child class
        self.model = None

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Fast initializer of a given layer (meant to be used with actor and critic)"""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        """Get return value of a given action in the model"""
        return self.critic(self.model(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.model(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    @abstractmethod
    def predict_proba(self, state):
        """
        Predict the probability of each action given the current state
        """
    @abstractmethod
    def predict(self, state):
        """Predict the best action at given state"""

    def predict_action_proba_at_state(self, state, action):
        """
        Predict the probability of the given action at the given state
        """
        return self.predict_proba(state)[action]

    def log_proba_action(self, state, action):
        """
        Log probability of the given action at the given state
        """
        return torch.log(self.predict_action_proba_at_state(state, action))

    @abstractmethod
    def optimize(self, *args):
        """
        Optimize the policy thanks to the given arguments
        """

    def play(self, horizon: int):
        """
        Lets the agent play in the environment according to the current policy, and outputs the score and transitions
        Parameter
        ---------
        horizon
            The number of episodes to play during this session
        """

        # We fetch the current state of the environment in which the policy is living
        current_state = self.environment.reset()[0]

        # Transitions stores the history of (states, actions, rewards) at each episode
        transitions = []

        # We go through the episodes, up to "horizon" episodes
        for t in range(horizon):
            # We predict the probability of each action being taken regarding the current state
            action_probabilities = self.predict_proba(current_state).data.numpy()

            # Now we draw an action with the "action_probabilities" density
            action = np.random.choice(np.array([0, 1]), p=action_probabilities)

            # We store the episode information into the transitions list
            # We add it as a clone so the upcoming update won't change what's being stored
            transitions.append((current_state.copy(), action, t + 1))

            # We compute the next state, having the environment making a step with the chosen action of the agent
            # Note that if "game_over" is True, it means that we've lost. So our model performs well if we do as
            # many episodes as possible.
            # Therefore, one score of the model is given by the length of the "transition" list.
            current_state, _, game_over, _, _ = self.environment.step(action)
            if game_over:
                break

        # See commentary above to under why the length of "transition" is a quality indicator to maximize
        score = len(transitions)
        return score, transitions
