from abc import ABC, abstractmethod
import numpy as np
from src.policies.abstract_policy import AbstractPolicy


class AbstractOptimizer(ABC):
    """
    Abstract class of the Optimizers

    This class is meant for programming purposes and can't be instantiated

    Parameters
    ----------
    policy
        Policy to optimize
    learning_rate
        A strictly positive float
    horizon
        Horizon of the discounted setting
    """

    def __init__(
            self,
            env,
            policy: AbstractPolicy,
            horizon: int,
            gamma: float,
    ):
        self.env = env  # a copy of the input environment
        self.policy = policy
        self.horizon = horizon
        self.gamma = gamma

    def train(self, max_trajectory, printEvery=-1) -> list[int]:

        scores = []

        for trajectory in range(max_trajectory):
            score = self.step()  # we do a training step
            scores.append(score)

            if printEvery > 0 and trajectory % printEvery == 0 and trajectory > 0:
                print('Trajectory {}\tAverage Score: {:.2f}'.format(trajectory, np.mean(scores[-50:-1])))

        return scores

    @abstractmethod
    def step(self) -> int:
        pass
