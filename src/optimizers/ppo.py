from abstract_optimizer import AbstractOptimizer
import numpy as np
from src.policies.abstract_policy import AbstractPolicy
from src.env.environment import Environment


class PPO(AbstractOptimizer):
    """
    Implementation of Proximal Policy Optimization
    """

    def __init__(
            self,
            environment: Environment,
            policy: AbstractPolicy,
            horizon: int,
            gamma: float
    ):
        super().__init__(environment, policy, horizon, gamma)

    def step(self) -> int:
        pass
