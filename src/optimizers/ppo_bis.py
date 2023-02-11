import sys

sys.path.append("C:/Users/lilia/OneDrive/Documents/GitHub/RL/src/env")

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
from src.env.environment import Environment
from src.policies.abstract_policy import AbstractPolicy
from collections import namedtuple
import ipdb

class PPO:
    def __init__(
        self,
        environment: Environment,
        learning_rate: float,
        horizon: int,
        actor_hidden: int,
        critic_hidden: int,
        clipping: float = 0.2,
        max_d_kl: float = 0.01,
        coef_entropy: float = 0,
        coef_value:float =0,
    ):

        self.environment = environment
        self.learning_rate = learning_rate
        self.horizon = horizon
        self.max_d_kl = max_d_kl
        self.clipping = clipping
        self.coef_entropy = coef_entropy
        self.coef_value = coef_value

        self.actor = nn.Sequential(
            nn.Linear(self.environment.observation_space.shape[0], actor_hidden),
            nn.ReLU(),
            nn.Linear(actor_hidden, self.environment.action_space.n),
            nn.Softmax(dim=1),
        )

        # Critic takes a state and returns its values
        self.critic = nn.Sequential(
            nn.Linear(self.environment.observation_space.shape[0], critic_hidden),
            nn.ReLU(),
            nn.Linear(critic_hidden, 1),
        )

        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.learning_rate)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.learning_rate)

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
        dist = Categorical(
            self.actor(state)
        )  # Create a distribution from probabilities for actions
        return dist.sample().item()

    def update_critic(self, advantages):
        loss = 0.5 * (advantages**2).mean()  # MSE
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def estimate_advantages(self, states, last_state, rewards):
        values = self.critic(states)
        last_value = self.critic(last_state.unsqueeze(0))
        next_values = torch.zeros_like(rewards)
        for i in reversed(range(rewards.shape[0])):
            last_value = next_values[i] = rewards[i] + 0.99 * last_value
        advantages = next_values - values
        ipdb.set_trace()
        return advantages

    def clipped_ratio(self, ratio):
        """Clipped ratio to avoid too big policy updates"""
        return torch.clamp(ratio, 1 - self.clipping, 1 + self.clipping)
        

    def compute_loss_actor(self, advantage, ratio, values, rewards, entropy=0):
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

    def update_actor(self, ratio, advantages, values, rewards, entropy):
        actor_loss = self.compute_loss_actor(advantages, ratio,values, rewards, entropy)  # MSE
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

    def update_agent(self, rollouts, rewards):
        states = torch.cat([r.states for r in rollouts], dim=0)
        actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()

        advantages = [
            self.estimate_advantages(states, next_states[-1], rewards)
            for states, _, rewards, next_states in rollouts
        ]
        advantages = torch.cat(advantages, dim=0).flatten()

        # Normalize advantages to reduce skewness and improve convergence
        advantages = (advantages - advantages.mean()) / advantages.std()

        

        distribution = self.actor(states)
        distribution = torch.distributions.utils.clamp_probs(distribution)
        probabilities = distribution[range(distribution.shape[0]), actions]

        # Now we have all the data we need for the algorithm

        # We will calculate the gradient wrt to the new probabilities (surrogate function),
        # so second probabilities should be treated as a constant
        # L = self.surrogate_loss(probabilities, probabilities.detach(), advantages)
        # KL = self.kl_div(distribution, distribution)
        probabilities_old = probabilities.detach()
        ratio = probabilities/probabilities_old
        values = entropy = torch.zeros(ratio.shape[0])
        try:
            self.update_actor(ratio, advantages, values, rewards,entropy)
        except:
            ipdb.set_trace()
        try:
            self.update_critic(advantages)
        except:
            ipdb.set_trace()

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
                    i+=1
                    with torch.no_grad():
                        action = self.get_action(state)

                    next_state, reward, done, _,_ = self.environment.step(action)

                    # Collect samples
                    samples.append((state, action, reward, next_state))

                    state = next_state

                    if i >= self.environment.max_duration:
                        done=True

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

            self.update_agent(rollouts, torch.Tensor(rewards))
            mtr = np.mean(rollout_total_rewards)
            print(
                f"E: {epoch}.\tMean total reward across {num_rollouts} rollouts: {mtr}"
            )

            mean_total_rewards.append(mtr)
        return mean_total_rewards


import gym
import sys

# sys.path.append("C:/Users/lilia/OneDrive/Documents/GitHub/RL/src/viz")
# import visualization as viz

# environment = gym.make("CartPole-v1")
# trpo = PPO(
#     environment=environment,
#     learning_rate=0.003,
#     horizon=150,
#     actor_hidden=64,
#     critic_hidden=64,
# )

# trpo_score = trpo.train(num_rollouts=5)


# viz.score_visualisation(np.array(trpo_score))
