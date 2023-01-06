from abstract_optimizer import AbstractOptimizer
import numpy as np
from src.policies.abstract_policy import AbstractPolicy


class PPO(AbstractOptimizer):
    """
    Implementation of Proximal Policy Optimization
    """

    def __init__(
            self,
            policy: AbstractPolicy,
            horizon: int,
            gamma: float
    ):
        super().__init__(policy, horizon, gamma)

    def step(self) -> int:
        pass
