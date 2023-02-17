import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
from src.env.environment import Environment
from collections import namedtuple

class TRPO():
    def __init__(self,
                 environment: Environment,
                 learning_rate: float,
                 horizon: int,
                 actor_hidden: int,
                 critic_hidden: int,
                 max_d_kl: float = 0.01):
        self.environment = environment
        self.learning_rate = learning_rate
        self.horizon = horizon
        self.max_d_kl = max_d_kl

        self.actor = nn.Sequential(nn.Linear(self.environment.observation_space_shape()[0], actor_hidden),
                                   nn.ReLU(),
                                   nn.Linear(actor_hidden, self.environment.n_actions),
                                   nn.Softmax(dim=1))

        # Critic takes a state and returns its values
        self.critic = nn.Sequential(nn.Linear(self.environment.observation_space_shape()[0], critic_hidden),
                                    nn.ReLU(),
                                    nn.Linear(critic_hidden, 1))

        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.learning_rate)

        self.Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states', ])

    def get_action(self, state):
        try: 
            state = torch.tensor(state).float().unsqueeze(0)  # Turn state into a batch with a single element
        except:
            state = torch.tensor(state[0]).float().unsqueeze(0)
        dist = Categorical(self.actor(state))  # Create a distribution from probabilities for actions
        return dist.sample().item()

    def update_critic(self, advantages):
        loss = .5 * (advantages ** 2).mean()  # MSE
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
        return advantages

    def surrogate_loss(self, new_probabilities, old_probabilities, advantages):
        return (new_probabilities / old_probabilities * advantages).mean()

    def kl_div(self, p, q):
        p = p.detach()
        return (p * (p.log() - q.log())).sum(-1).mean()

    def flat_grad(self, y, x, retain_graph=False, create_graph=False):
        if create_graph:
            retain_graph = True

        g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        g = torch.cat([t.view(-1) for t in g])
        return g

    def conjugate_gradient(self, A, b, delta=0., max_iterations=10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()

        i = 0
        while i < max_iterations:
            AVP = A(p)

            dot_old = r @ r
            alpha = dot_old / (p @ AVP)

            x_new = x + alpha * p

            if (x - x_new).norm() <= delta:
                return x_new

            i += 1
            r = r - alpha * AVP

            beta = (r @ r) / dot_old
            p = r + beta * p

            x = x_new
        return x

    def apply_update(self, grad_flattened):
        n = 0
        for p in self.actor.parameters():
            numel = p.numel()
            g = grad_flattened[n:n + numel].view(p.shape)
            p.data += g
            n += numel

    def update_agent(self, rollouts):
        states = torch.cat([r.states for r in rollouts], dim=0)
        actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()

        advantages = [self.estimate_advantages(states, next_states[-1], rewards) for states, _, rewards, next_states in
                      rollouts]
        advantages = torch.cat(advantages, dim=0).flatten()

        # Normalize advantages to reduce skewness and improve convergence
        advantages = (advantages - advantages.mean()) / advantages.std()

        self.update_critic(advantages)

        distribution = self.actor(states)
        distribution = torch.distributions.utils.clamp_probs(distribution)
        probabilities = distribution[range(distribution.shape[0]), actions]

        # Now we have all the data we need for the algorithm

        # We will calculate the gradient wrt to the new probabilities (surrogate function),
        # so second probabilities should be treated as a constant
        L = self.surrogate_loss(probabilities, probabilities.detach(), advantages)
        KL = self.kl_div(distribution, distribution)

        parameters = list(self.actor.parameters())

        g = self.flat_grad(L, parameters, retain_graph=True)
        d_kl = self.flat_grad(KL, parameters,
                              create_graph=True)  # Create graph, because we will call backward() on it (for HVP)

        def HVP(v):
            return self.flat_grad(d_kl @ v, parameters, retain_graph=True)

        search_dir = self.conjugate_gradient(HVP, g)
        max_length = torch.sqrt(2 * self.max_d_kl / (search_dir @ HVP(search_dir)))
        max_step = max_length * search_dir

        def criterion(step):
            self.apply_update(step)

            with torch.no_grad():
                distribution_new = self.actor(states)
                distribution_new = torch.distributions.utils.clamp_probs(distribution_new)
                probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]

                L_new = self.surrogate_loss(probabilities_new, probabilities, advantages)
                KL_new = self.kl_div(distribution, distribution_new)

            L_improvement = L_new - L

            if L_improvement > 0 and KL_new <= self.max_d_kl:
                return True

            self.apply_update(-step)
            return False

        i = 0
        while not criterion((0.9 ** i) * max_step) and i < 10:
            i += 1

    def train(self, num_rollouts=10):
        mean_total_rewards = []
        global_rollout = 0

        for epoch in range(self.horizon):
            rollouts = []
            rollout_total_rewards = []

            for t in range(num_rollouts):
                state, _ = self.environment.reset()
                done = False

                samples = []

                i = 0
                while not done:
                    i += 1
                    with torch.no_grad():
                        action = self.get_action(state)

                    next_state, reward, done, _, _ = self.environment.step(action)

                    # Collect samples
                    samples.append((state, action, reward, next_state))

                    state = next_state

                    if i >= self.environment.max_duration:
                        done = True

                # Transpose our samples
                states, actions, rewards, next_states = zip(*samples)

                states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
                next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
                actions = torch.as_tensor(actions).unsqueeze(1)
                rewards = torch.as_tensor(rewards).unsqueeze(1)

                rollouts.append(self.Rollout(states, actions, rewards, next_states))

                rollout_total_rewards.append(rewards.sum().item())
                global_rollout += 1
            print(f'rollout phase for epoch {epoch} done')
            self.update_agent(rollouts)
            mtr = np.mean(rollout_total_rewards)
            print(f'E: {epoch}.\tMean total reward across {num_rollouts} rollouts: {mtr}')
            if mtr > 3000:
                ipdb.set_trace()
            mean_total_rewards.append(mtr)
        return mean_total_rewards
