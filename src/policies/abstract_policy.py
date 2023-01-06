from abc import ABC, abstractmethod
from src.env.environment import Environment
import numpy as np


class AbstractPolicy(ABC):
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

    @abstractmethod
    def predict_proba(self, state):
        """
        Predict the probability of each action given the current state
        """

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
