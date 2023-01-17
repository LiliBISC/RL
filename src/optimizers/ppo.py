from abstract_optimizer import AbstractOptimizer
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
            c1: float,
            c2: float,
            clipping: float = 0.2,
    ):
        super().__init__(policy, horizon, gamma)
        self.old_policy = policy
        self.clipping = clipping
        self.lmbda = lmbda
        self.c1 = c1
        self.c2 = c2

    def ratio(self, a_t, s_t):
        """Probability ratio of taking action (a_t) at state (s_t) regarding the old policy

        If the ratio > 1, then the (a_t) at (s_t) is more likely in the current policy than the old one
        Therefore, the ratio estimates the divergence of the new policy compared to the old one
        """
        # Fetching the probability of the given action regarding the current state
        proba_current = self.policy.predict_action_proba_at_state(s_t, a_t)
        proba_old = self.old_policy.predict_action_proba_at_state(s_t, a_t)

        return proba_current / proba_old

    def clipped_ratio(self, ratio):
        """Clipped ratio to avoid too big policy updates"""
        return torch.clamp(ratio, 1 - self.clipping, 1 + self.clipping)

    def should_update_policy(self, a_t, s_t, advantage):
        """Whether we should update the policy or not, based on the ratio"""
        # We compute the ratio
        r_t = self.ratio(a_t, s_t)

        # We update the policy iif:
        # - we're in range if the clipped surrogate objective [1 - eps, 1 + eps]
        # - the advantage leads to getting closer to the range of the surrogate objective
        #   * Being below clipped ratio but advantage > 0
        #   * Being above clipped ratio but advantage < 0
        if 1 - self.clipping <= r_t <= 1 + self.clipping:
            return True
        if r_t >= 1 + self.clipping and advantage < 0:
            return True
        if r_t <= 1 - self.clipping and advantage > 0:
            return True
        return False

    def compute_loss(self, advantage, ratio,
                     values, targets,
                     entropy,
                     c1, c2):
        """Compute the overall loss of the PPO"""
        # Policy loss
        loss_clip = torch.max(
            -advantage * ratio
            - advantage * self.clipped_ratio(ratio)
        ).mean()
        # Entropy loss
        loss_entropy = entropy.mean()
        # Value loss (unclipped)
        loss_value = 0.5 * ((values - targets) ** 2).mean()
        return loss_clip - c1 * loss_value + c2 * loss_entropy

    def step(self) -> int:  # This should output a score
        # ALGO Logic: Storage setup
        # horizon step, 1 environment
        obs = torch.zeros((self.horizon, 1) + self.policy.environment.observation_space_shape())
        actions = torch.zeros((self.horizon, 1) + self.policy.environment.action_space_shape())
        log_probas = torch.zeros((self.horizon, 1))
        rewards = torch.zeros((self.horizon, 1))
        dones = torch.zeros((self.horizon, 1))
        values = torch.zeros((self.horizon, 1))

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        next_obs = torch.Tensor(self.policy.environment.reset())
        next_done = torch.zeros(1)

        for t in range(0, self.horizon):
            global_step += 1
            obs[t] = next_obs
            dones[t] = next_done

            with torch.no_grad():
                action, log_proba, _, value = self.policy.get_action_and_value(next_obs)
                values[t] = value.flatten()
            actions[t] = action
            log_probas[t] = log_proba

            next_obs, reward, done, info = self.policy.environment.step(action.cpu().numpy())
            rewards[t] = torch.tensor(reward).view(-1)
            next_obs, next_done = torch.Tensor(next_obs), torch.Tensor(done)

            with torch.no_grad():
                next_value = self.policy.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for k in reversed(range(self.horizon)):
                    if k == self.horizon - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[k + 1]
                        nextvalues = values[k + 1]
                    delta = rewards[k] + self.gamma * nextvalues * nextnonterminal - values[k]
                    advantages[k] = lastgaelam = delta + self.gamma * self.lmbda * nextnonterminal * lastgaelam
                returns = advantages + values

                obs = obs.reshape((-1,) + self.policy.environment.observation_space_shape())
                log_probas = log_probas.reshape(-1)
                actions = actions.reshape((-1,) + self.policy.environment.action_space_shape())
                advantages = advantages.reshape(-1)
                returns = returns.reshape(-1)
                values = values.reshape(-1)

                # That's when PPO is actually stepping, before that was just precomputations
                _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(obs, actions.long())
                logratio = newlogprob - log_probas
                ratio = logratio.exp()

                loss = self.compute_loss(advantages, ratio, values, entropy, returns, self.c1, self.c2)

                self.policy.optimize(loss)

        score, _ = self.policy.play(self.horizon)
        return score
