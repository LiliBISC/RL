from abc import ABC, abstractmethod
import numpy as np
import sys
from src.policies.abstract_policy import AbstractPolicy
from src.env.environment import Environment


class AbstractOptimizer(ABC):
    """
    Abstract class of the Optimizers

    This class is meant for programming purposes and can't be instantiated

    Parameters
    ----------
    policy
        Policy to optimize (already evolving in a given environment)
    horizon
        Horizon of the discounted setting
    gamma
        Discounted setting parameter
    """

    def __init__(
            self,
            policy: AbstractPolicy,
            horizon: int,
            gamma: float,
    ):
        self.policy = policy
        self.horizon = horizon
        self.gamma = gamma

    def train(self, max_trajectory, printEvery=-1) -> np.array:
        """
        Train the policy on the environment using the optimizer
        """
        # Initializing the scores output
        scores = np.zeros(max_trajectory)

        # We run "max_trajectory" experiments and compute the score for each of them to see model improvement
        for trajectory in range(max_trajectory):
            # We do an optimization step of the model
            score = self.step()
            # Storing the output score
            scores[trajectory] = score

            if printEvery > 0 and trajectory % printEvery == 0 and trajectory > 0:
                print('Trajectory {}\tAverage Score: {:.2f}'.format(trajectory, np.mean(
                    scores[trajectory - printEvery:trajectory])))
            
            print(f"trajectory {trajectory} is done")

        return scores

    @abstractmethod
    def step(self) -> int:
        """
        Apply a step of optimization of the optimizer on the policy with the given environment
        """
        raise Exception("Step function is not implemented in child class")
