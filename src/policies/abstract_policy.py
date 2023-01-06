from abc import ABC, abstractmethod


class AbstractPolicy(ABC):
    """
    Abstract class of any policy

    This class is meant for programming purposes and can't be instantiated
    """

    def __init__(self, env):
        self.env = env
        self.obs_size = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

    @abstractmethod
    def predict_proba(self, state):
        """
        Predict the probability of the given state
        """

    @abstractmethod
    def learn_one(self, transitions, gamma):
        """
        Learns the transitions with the given gamma
        """