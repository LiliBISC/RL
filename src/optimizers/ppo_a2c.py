import sys
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from src.env.environment import Environment
from src.policies.abstract_policy import AbstractPolicy
from collections import namedtuple
import ipdb


class PPO_A2C(nn.Module):
    """
    Implementation of ActorCritics
    """

    def __init__(
            self, 
            environment : Environment, 
            learning_rate : float, 
            horizon : int,
            clipping: float =0.2,
            max_d_kl: float=0.01,
            coef_entropy: float=0,
            coef_values: float=0
    ): 
        super(PPO_A2C, self).__init__()
        self.environment = environment
        self.learning_rate = learning_rate
        self.horizon = horizon
        self.clipping = clipping
        self.max_d_kl = max_d_kl
        self.coef_entropy = coef_entropy
        self.coef_values = coef_values

        self.linear1 = nn.Linear(self.environment.observation_space.shape[0], 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        
        self.actor = nn.Linear(64, self.environment.action_space.n)
        self.critic = nn.Linear(64, 1)


    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        
        x = self.linear2(x)
        x = F.relu(x)
        
        x = self.linear3(x)
        x = F.relu(x)

        return x
    
    def get_action_probs(self, x):
        x = self(x)
        action_probs = F.softmax(self.actor(x), dim = -1)
        return action_probs
    
    def evaluate_actions(self, x):
        x = self(x)
        action_probs = F.softmax(self.actor(x), dim = -1)
        state_values = self.critic(x)
        return action_probs, state_values

    def test_model(self):
        score = 0
        done = False
        state = self.environment.reset()
        global action_probs
        while not done:
            if type(state) == tuple:
                state=state[0]
            score += 1
            s = torch.from_numpy(state).float().unsqueeze(0)
            
            action_probs = self.get_action_probs(Variable(s))
            
            _, action_index = action_probs.max(1)
            action = action_index.data[0] 
            next_state, reward, done, thing, _ = self.environment.step(int(action))
            state = next_state
            
        return score

    def clipped_ratio(self,ratio):
        return torch.clamp(ratio, 1-self.clipping, 1+self.clipping)
    
    def compute_loss(self, ratio, advantages, rewards, values, entropy):
        loss_clip = torch.min(
            advantages * ratio,
            advantages * self.clipped_ratio(ratio),
        ).mean()

        # Entropy loss
        loss_entropy = entropy.mean()

        # Value loss
        loss_value = 0.5 * ((values - rewards) ** 2).mean()
        # Total loss
        return -(
            loss_clip - self.coef_values * loss_value + self.coef_entropy * loss_entropy
        )

    def train_test(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        scores = []
        states = []
        actions = []
        rewards = []

        for i in range(self.horizon):
            
            del states[:]
            del actions[:]
            del rewards[:]
            
            state = self.environment.reset() 
            done = False
            
            # act phase
            while not done:
                if type(state) == tuple:
                    state = state[0]
                s = torch.Tensor(state).float().unsqueeze(0)
                action_probs = self.get_action_probs(Variable(s))
                try:
                    action = action_probs.multinomial(num_samples=1).data[0][0]
                except:
                    ipdb.set_trace()
                next_state, reward, done, _, _ = self.environment.step(int(action))
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                state = next_state

            if len(rewards) < 200: # only reflecting/training on episodes where a failure occured. No training
                # signal in perfect games. 
                # Reflect phase
                if i%50 == 0:
                    print("Training : ", i)
                    print("         * Score : ", len(rewards))

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

                global state_values
                action_probs, state_values = self.evaluate_actions(s)
                state_values = torch.Tensor(state_values)
                action_probs_old = action_probs.detach()
                action_log_probs = action_probs.log() 
                ratio = action_probs/action_probs_old
                advantages = Variable(torch.FloatTensor(np.array(rewards))).unsqueeze(1) - state_values
                entropy = (action_probs * action_log_probs).sum(1).mean()
                loss = self.compute_loss(
                    ratio, 
                    advantages, 
                    torch.Tensor(rewards), 
                    state_values,
                    entropy
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                optimizer.step()
                        
            else: 
                if i%10 == 0:
                    print("Not Training : ", i)
                    print("         * Score : ", len(rewards))
            
            s = self.test_model()
            scores.append(s)

        return np.array(scores)

