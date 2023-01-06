from abc import ABC, abstractmethod
from src.env.environment import Environment
import numpy as np


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
    def predict_proba(self, state: np.array) -> np.array:
        """
        Predict the probability of each action given the current state
        """

    @abstractmethod
    def learn_one(self, transitions: list, gamma: float):
        """
        Learns the transitions with the given discounted parameter gamma
        """
