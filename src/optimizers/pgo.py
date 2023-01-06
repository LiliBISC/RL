from .abstract_optimizer import AbstractOptimizer
import numpy as np
from src.policies.abstract_policy import AbstractPolicy


class PGO(AbstractOptimizer):
    """
    Implementation of Proximal Policy Optimization
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
        curr_state = self.env.reset()[0]
        done = False
        transitions = []

        for t in range(self.horizon):
            act_prob = self.policy.predict_proba(curr_state)
            action = np.random.choice(np.array([0, 1]), p=act_prob.data.numpy())
            prev_state = curr_state
            curr_state, _, done, info, _ = self.env.step(action)
            transitions.append((prev_state, action, t + 1))
            if done:
                break

        score = len(transitions)

        self.policy.learn_one(transitions, self.gamma)
        return score