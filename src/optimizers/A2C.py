import sys
sys.path.append('C:/Users/lilia/OneDrive/Documents/GitHub/RL/src/env')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from environment import Environment

class A2C(nn.Module):
    """
    Implementation of ActorCritics
    """

    def __init__(self, environment : Environment, learning_rate : float, horizon : int): 
        super(A2C, self).__init__()
        self.environment = environment
        self.learning_rate = learning_rate
        self.horizon = horizon

        self.linear1 = nn.Linear(self.environment.n_observations, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        
        self.actor = nn.Linear(64, self.environment.n_actions)
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

    def train_test(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

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
                s = torch.from_numpy(state).float().unsqueeze(0)
                action_probs = self.get_action_probs(Variable(s))
                action = action_probs.multinomial(num_samples=1).data[0][0]
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

                action_log_probs = action_probs.log() 

                advantages = Variable(torch.FloatTensor(rewards)).unsqueeze(1) - state_values

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
                        
            else: 
                if i%50 == 0:
                    print("Not Training : ", i)
                    print("         * Score : ", len(rewards))
            
            s = self.test_model()
            scores.append(s)

        return np.array(scores)
