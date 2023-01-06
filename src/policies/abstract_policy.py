from abc import ABC, abstractmethod
from src.env.environment import Environment


class AbstractPolicy(ABC):
    """
    Abstract class of any policy

    This class is meant for programming purposes and can't be instantiated
    """

    def __init__(
            self,
            environment: Environment
    ):
        self.environment = environment

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
