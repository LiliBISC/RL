from abstract_optimizer import AbstractOptimizer
from src.policies.abstract_policy import AbstractPolicy


class TRPO(AbstractOptimizer):
    """
    Implementation of Trust Region Policy Optimization
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
