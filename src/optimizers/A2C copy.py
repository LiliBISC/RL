import sys
sys.path.append('C:/Users/lilia/OneDrive/Documents/GitHub/RL/src/env')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Adam
from torch.distributions import Categorical
from torch.autograd import Variable
from environment import Environment
from collections import namedtuple

class A2C():
    """
    Implementation of ActorCritics
    """

    def __init__(self,
                environment : Environment,
                learning_rate : float,
                horizon : int,
                actor_hidden : int, 
                critic_hidden : int): 
        self.environment = environment
        self.learning_rate = learning_rate
        self.horizon = horizon

        self.actor = nn.Sequential(nn.Linear(self.environment.n_observations, actor_hidden),
                                nn.ReLU(),
                                nn.Linear(actor_hidden, self.environment.n_actions),
                                nn.Softmax(dim=1))

        # Critic takes a state and returns its values
        self.critic = nn.Sequential(nn.Linear(self.environment.n_observations, critic_hidden),
                                nn.ReLU(),
                                nn.Linear(critic_hidden, 1))

        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.learning_rate)

        self.Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states', ])
    
    def get_action_probs(self, state):
        state = torch.tensor(state).float().unsqueeze(0)  # Turn state into a batch with a single element
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
    
    def evaluate_actions(self, x):
        x = self(x)
        action_probs = F.softmax(self.actor(x), dim = -1)
        state_values = self.critic(x)
        return action_probs, state_values

    def update_agent(self, rollout): 
        states = [r.states for r in rollout]
        rewards = [r.rewards for r in rollout]
        actions = [r.actions for r in rollout]

        advantages = np.array([self.estimate_advantages(states, next_states[-1], rewards) for states, _, rewards, next_states in rollout])
        # Normalize advantages to reduce skewness and improve convergence
        advantages = (advantages - advantages.mean()) / advantages.std()

        self.update_critic(advantages)

        R = []
        rr = rewards
        rr.reverse()

        next_return = -30 

        for r in range(len(rr)):
            this_return = rr[r] + next_return * .9
            R.append(this_return)
            next_return = this_return
        R.reverse()

        rewards = R
        
        # taking only the last 20 states before failure
        rewards = rewards[-20:]
        states = states[-20:]
        actions = actions[-20:]
        
        s = Variable(torch.FloatTensor(states))

        action_probs, state_values = self.evaluate_actions(s)

        action_log_probs = action_probs.log() 

        entropy = (action_probs * action_log_probs).sum(1).mean()

        a = Variable(torch.LongTensor(actions).view(-1,1))

        chosen_action_log_probs = action_log_probs.gather(1, a)

        action_gain = (chosen_action_log_probs * advantages).mean()

        value_loss = advantages.pow(2).mean()

        total_loss = value_loss - action_gain - 0.0001*entropy

        optimizer.zero_grad()

        total_loss.backward()

        nn.utils.clip_grad_norm_(self.parameters(), 0.5)

        optimizer.step()
        print('ok')

    def train_test(self):
        scores = []
        for i in range(self.horizon):
            
            
            rollouts = []
            rollout_total_rewards = []

            state = self.environment.reset() 
            done = False

            samples = []
            # act phase
            while not done:
                with torch.no_grad():
                        action = self.get_action(state)

                next_state, reward, done, _ = self.environment.step(action)

                # Collect samples
                samples.append((state, action, reward, next_state))

                state = next_state

            # Transpose our samples
            states, actions, rewards, next_states = zip(*samples)

            states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
            next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
            actions = torch.as_tensor(actions).unsqueeze(1)
            rewards = torch.as_tensor(rewards).unsqueeze(1)

            rollouts.append(self.Rollout(states, actions, rewards, next_states))

            rollout_total_rewards.append(rewards.sum().item())

            scores.append(rollout_total_rewards)
            self.update_agent(rollouts)

        return np.array(scores)
