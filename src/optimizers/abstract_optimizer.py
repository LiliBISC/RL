from abc import ABC, abstractmethod
import numpy as np
from src.policies.abstract_policy import AbstractPolicy
from src.env.environment import Environment


class AbstractOptimizer(ABC):
    """
    Abstract class of the Optimizers

    This class is meant for programming purposes and can't be instantiated

    Parameters
    ----------
    environment
        Environment in which to train the model
    policy
        Policy to optimize
    horizon
        Horizon of the discounted setting
    gamma
        Discounted setting parameter
    """

    def __init__(
            self,
            environment: Environment,
            policy: AbstractPolicy,
            horizon: int,
            gamma: float,
    ):
        self.environment = environment  # a copy of the input environment
        self.policy = policy
        self.horizon = horizon
        self.gamma = gamma

    def train(self, max_trajectory, printEvery=-1) -> np.array:
        """
        Train the policy on the environment using the optimizer
        """
        scores = np.zeros(max_trajectory)

        for trajectory in range(max_trajectory):
            score = self.step()  # we do a training step
            scores[trajectory] = score

            if printEvery > 0 and trajectory % printEvery == 0 and trajectory > 0:
                print('Trajectory {}\tAverage Score: {:.2f}'.format(trajectory, np.mean(
                    scores[trajectory - printEvery:trajectory])))

        return scores

    @abstractmethod
    def step(self) -> int:
        pass
