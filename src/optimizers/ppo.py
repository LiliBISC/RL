import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
from src.env.environment import Environment
from collections import namedtuple


class _A2C_PPO_NN(nn.Module):
    """
    Internal class to implement the A2C architecture into the PPO
    """

    def __init__(
            self,
            input_space: int,
            actor_space: int,
    ) -> None:
        super(_A2C_PPO_NN, self).__init__()
        self.linear1 = nn.Linear(input_space, 64)
        self.linear2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, actor_space)
        self.critic = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        action_probas = self.softmax(self.actor(x))
        state_values = self.critic(x)
        return action_probas, state_values


class PPO:
    """
    Implementation in Actor-Critic + Entropy style of the PPO algorithm

    Parameter
    ---------
    environment
        Environment in which to train
    learning_rate
        Learning rate of the optimizer of the agent
    horizon
        Number of episodes to be played
    clipping
        Clipping parameter of the ratio
    coef_entropy
        Importance coefficient of the entropy
    coef_value
        Importance coefficient of the critic
    """

    def __init__(
            self,
            environment: Environment,
            learning_rate: float,
            horizon: int,
            clipping: float = 0.2,
            max_d_kl: float = 0.01,
            coef_entropy: float = 0,
            coef_value: float = 0,
    ):

        self.environment = environment
        self.learning_rate = learning_rate
        self.horizon = horizon
        self.max_d_kl = max_d_kl
        self.clipping = clipping
        self.coef_entropy = coef_entropy
        self.coef_value = coef_value

        self.model = _A2C_PPO_NN(
            self.environment.n_observations,
            self.environment.n_actions
        )
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        self.Rollout = namedtuple(
            "Rollout",
            [
                "states",
                "actions",
                "rewards",
                "next_states",
            ],
        )

    def get_action(self, state):
        state = (
            torch.tensor(state).float().unsqueeze(0)
        )  # Turn state into a batch with a single element
        output_actor, _ = self.model(state)
        dist = Categorical(
            output_actor
        )  # Create a distribution from probabilities for actions
        return dist.sample().item()

    def estimate_advantages(self, states, last_state, rewards):
        _, values = self.model(states)
        _, last_value = self.model(last_state)
        next_values = torch.zeros_like(rewards)
        for i in reversed(range(rewards.shape[0])):
            last_value = next_values[i] = rewards[i] + 0.99 * last_value
        advantages = next_values - values
        # ipdb.set_trace()
        return advantages

    def clipped_ratio(self, ratio):
        """Clipped ratio to avoid too big policy updates"""
        return torch.clamp(ratio, 1 - self.clipping, 1 + self.clipping)

    def compute_loss(self, advantage, ratio, values, rewards, entropy):
        loss_clip = torch.min(
            advantage * ratio,
            advantage * self.clipped_ratio(ratio),
        ).mean()

        # Entropy loss
        loss_entropy = entropy.mean()

        # Value loss
        loss_value = 0.5 * ((values - rewards) ** 2).mean()
        # Total loss
        return -(
                loss_clip - self.coef_value * loss_value + self.coef_entropy * loss_entropy
        )

    def update_network(self, ratio, advantages, values, rewards, entropy):
        loss = self.compute_loss(advantages, ratio, values, rewards, entropy)  # MSE
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def update_agent(self, rollouts):
        states = torch.cat([r.states for r in rollouts], dim=0)
        actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()
        actor_distribution, state_values = self.model(states)
        advantages = [
            self.estimate_advantages(states, next_states[-1], rewards)
            for states, _, rewards, next_states in rollouts
        ]
        advantages = torch.cat(advantages, dim=0).flatten()
        rewards = [rewards for _, _, rewards, _ in rollouts]
        rewards = torch.cat(rewards, dim=0).flatten()
        # Normalize advantages to reduce skewness and improve convergence
        advantages = (advantages - advantages.mean()) / advantages.std()

        distribution = actor_distribution
        distribution = torch.distributions.utils.clamp_probs(distribution)
        probabilities = distribution[range(distribution.shape[0]), actions]

        # Now we have all the data we need for the algorithm

        # We will calculate the gradient wrt to the new probabilities (surrogate function),
        # so second probabilities should be treated as a constant
        probabilities_old = probabilities.detach()
        ratio = probabilities / probabilities_old
        action_log_probs = actor_distribution.log()
        entropy = (actor_distribution * action_log_probs).sum(1).mean()
        self.update_network(ratio, advantages, state_values, rewards, entropy)

    def train(self, num_rollouts=10):
        mean_total_rewards = []
        global_rollout = 0

        for epoch in range(self.horizon):
            rollouts = []
            rollout_total_rewards = []

            for t in range(num_rollouts):
                state = self.environment.reset()[0]
                done = False

                samples = []

                i = 0
                while not done:
                    i += 1
                    with torch.no_grad():
                        action = self.get_action(state)
                    try:
                        next_state, reward, done, _, _ = self.environment.step(action)
                    except:
                        actions = np.zeros(self.environment.n_actions)
                        actions[action] = 1
                        next_state, reward, done, _, _ = self.environment.step(actions)

                    # Collect samples
                    samples.append((state, action, reward, next_state))

                    state = next_state

                    if i >= self.environment.max_duration:
                        done = True

                # Transpose our samples
                states, actions, rewards, next_states = zip(*samples)

                states = torch.stack(
                    [torch.from_numpy(state) for state in states], dim=0
                ).float()
                next_states = torch.stack(
                    [torch.from_numpy(state) for state in next_states], dim=0
                ).float()
                actions = torch.as_tensor(actions).unsqueeze(1)
                rewards = torch.as_tensor(rewards).unsqueeze(1)

                rollouts.append(self.Rollout(states, actions, rewards, next_states))

                rollout_total_rewards.append(rewards.sum().item())
                global_rollout += 1

            self.update_agent(rollouts)
            mtr = np.mean(rollout_total_rewards)
            print(
                f"E: {epoch}.\tMean total reward across {num_rollouts} rollouts: {mtr}"
            )

            mean_total_rewards.append(mtr)
        return mean_total_rewards
