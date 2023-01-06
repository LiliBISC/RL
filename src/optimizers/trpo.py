from abstract_optimizer import AbstractOptimizer
from src.policies.abstract_policy import AbstractPolicy


class TRPO(AbstractOptimizer):
    """
    Implementation of Trust Region Policy Optimization
    """

    def __init__(
            self,
            env,
            policy: AbstractPolicy,
            horizon: int,
            gamma: float
    ):
        super().__init__(env, policy, horizon, gamma)

    def step(self) -> int:
        pass
