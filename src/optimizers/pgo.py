from .abstract_optimizer import AbstractOptimizer
import numpy as np
from src.policies.abstract_policy import AbstractPolicy
from src.env.environment import Environment


class PGO(AbstractOptimizer):
    """
    Implementation of Vanilla Policy Gradient Optimizer

    Parameter
    ---------
    policy
        Policy to optimize (living in a given environment)
    horizon
        Horizon of the discounted setting
    gamma
        Discounted setting parameter
    """

    def __init__(
            self,
            policy: AbstractPolicy,
            horizon: int,
            gamma: float
    ):
        super().__init__(policy, horizon, gamma)

    def step(self) -> int:
        # We fetch the current state of the environment in which the policy is living
        current_state = self.policy.environment.reset()[0]

        # Transitions stores the history of (states, actions, rewards) at each episode
        transitions = []

        # We go through the episodes, up to "horizon" episodes
        for t in range(self.horizon):
            # We predict the probability of each action being taken regarding the current state
            action_probabilities = self.policy.predict_proba(current_state)

            # Now we draw an action with the "action_probabilities" density
            action = np.random.choice(np.array([0, 1]), p=action_probabilities)

            # We store the episode information into the transitions list
            # We add it as a clone so the upcoming update won't change what's being stored
            transitions.append((current_state.copy(), action, t + 1))

            # We compute the next state, having the environment making a step with the chosen action of the agent
            # Note that if "game_over" is True, it means that we've lost. So our model performs well if we do as
            # many episodes as possible.
            # Therefore, one score of the model is given by the length of the "transition" list.
            current_state, _, game_over, _, _ = self.policy.environment.step(action)
            if game_over:
                break

        # See commentary above to under why the length of "transition" is a quality indicator to maximize
        score = len(transitions)

        # Now we improve the policy with the transitions experience,
        # with the discounted setting parameter of the problem
        self.policy.learn_one(transitions, self.gamma)
        return score
