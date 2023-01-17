from src.optimizers.abstract_optimizer import AbstractOptimizer
import torch
from src.policies.abstract_policy import AbstractPolicy


class PPO(AbstractOptimizer):
    """
    Implementation of Proximal Policy Optimization

    Parameter
    ---------
    policy
        Policy to optimize (living in a given environment)
    horizon
        Horizon of the discounted setting
    gamma
        Discounted setting parameter
    clipping
        Clipping parameter of the proximal policy (0.2 in the paper)
    """

    def __init__(
            self,
            policy: AbstractPolicy,
            horizon: int,
            gamma: float,
            lmbda: float,
            coef_entropy: float,
            coef_value: float,
            clipping: float = 0.2,
    ):
        super().__init__(policy, horizon, gamma)
        self.old_policy = policy
        self.clipping = clipping
        self.lmbda = lmbda
        self.coef_entropy = coef_entropy
        self.coef_value = coef_value

    def clipped_ratio(self, ratio):
        """Clipped ratio to avoid too big policy updates"""
        return torch.clamp(ratio, 1 - self.clipping, 1 + self.clipping)

    def compute_loss(self, advantage, ratio,
                     values, returns,
                     entropy):
        """Compute the overall loss of the PPO"""
        # Policy loss (clipped)
        loss_clip = torch.max(
            - advantage * ratio
            - advantage * self.clipped_ratio(ratio)
        ).mean()

        # Entropy loss
        loss_entropy = entropy.mean()

        # Value loss
        loss_value = 0.5 * ((values - returns) ** 2).mean()
        # Total loss
        return loss_clip - self.coef_value * loss_value + self.coef_entropy * loss_entropy

    def step(self) -> int:
        # Storage setup
        # horizon steps to do, 1 environment
        states = torch.zeros((self.horizon, 1) + self.policy.environment.observation_space_shape())
        actions = torch.zeros((self.horizon, 1) + self.policy.environment.action_space_shape())
        log_probas = torch.zeros((self.horizon, 1))
        rewards = torch.zeros((self.horizon, 1))
        gameovers = torch.zeros((self.horizon, 1))
        values = torch.zeros((self.horizon, 1))

        global_step = 0
        # Fetching the current default state of the environment
        state = torch.Tensor(self.policy.environment.reset()[0])
        # Initializing the gameover situation as a zero (not gameover)
        gameover = torch.zeros(1)

        for t in range(0, self.horizon):
            global_step += 1
            # Saving the current state of the environment
            states[t] = state
            # Saving the gameover memory
            gameovers[t] = gameover

            # Predicting the action to do at time "t" regarding the current state
            with torch.no_grad():
                # We also output the log probability of the action, and the value function output
                action, log_proba, _, value = self.policy.get_action_and_value(state)
                values[t] = value.flatten()
            # Saving the chosen action
            actions[t] = action
            # Saving the log probability of the chosen action
            log_probas[t] = log_proba

            # Let the agent play for a step and get the results
            state, reward, gameover, info, _ = self.policy.environment.step(action.cpu().numpy())
            # Saving reward
            rewards[t] = torch.tensor(reward).view(-1)
            # Making observation and gameover tensor-like for computations
            state, gameover = torch.Tensor(state), torch.Tensor([gameover])

            # Computing the return & advantages
            # This is still to be understood boyzz
            with torch.no_grad():
                value = self.policy.get_value(state).reshape(1, -1)
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for k in reversed(range(self.horizon)):
                    if k == self.horizon - 1:
                        next_non_terminal = 1.0 - gameover
                        next_values = value
                    else:
                        next_non_terminal = 1.0 - gameovers[k + 1]
                        next_values = values[k + 1]
                    delta = rewards[k] + self.gamma * next_values * next_non_terminal - values[k]
                    advantages[k] = lastgaelam = delta + self.gamma * self.lmbda * next_non_terminal * lastgaelam
                returns = advantages + values

            # Reshapes for computation
            states_flat = states.reshape((-1,) + self.policy.environment.observation_space_shape())
            log_probas_flat = log_probas.reshape(-1)
            actions_flat = actions.reshape((-1,) + self.policy.environment.action_space_shape())

            # And this is when PPO is actually stepping
            # We compute the proba the actions regarding the states
            _, next_log_proba, entropy, next_value = self.policy.get_action_and_value(states_flat, actions_flat.long())

            # Computing the ratio
            log_ratio = next_log_proba - log_probas_flat
            ratio = log_ratio.exp()

            # Computing loss
            loss = self.compute_loss(
                advantages.reshape(-1),
                ratio,
                values.reshape(-1),
                entropy,
                returns.reshape(-1)
            )

            # Optimizing the policy with the computed loss
            self.policy.optimize(loss)

        # And since we would like to output a score, we let the agent play into the environment with the
        # newly optimized policy
        score, _ = self.policy.play(self.horizon)
        return score
