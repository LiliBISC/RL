from abstract_optimizer import AbstractOptimizer
from src.policies.abstract_policy import AbstractPolicy
from src.env.environment import Environment


class TRPO(AbstractOptimizer):
    """
    Implementation of Trust Region Policy Optimization
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
