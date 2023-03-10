import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.env.environment import Environment
from torch.autograd import Variable


class A2C(nn.Module):
    """
    Implementation of Actor-Critic architecture

    Parameter
    ---------
    environment
        Environment in which to train
    learning_rate
        Learning rate of the optimizer of the agent
    horizon
        Number of episodes to be played
    actor_hidden
        Size of the hidden network of the actor
    critic_hidden
        Size of the hidden network of the critic
    """

    def __init__(self,
                 environment: Environment,
                 learning_rate: float,
                 horizon: int,
                 actor_hidden: int = 64,
                 critic_hidden: int = 64):
        super(A2C, self).__init__()
        self.environment = environment
        self.learning_rate = learning_rate
        self.horizon = horizon

        self.linear1 = nn.Linear(self.environment.observation_space_shape()[0], actor_hidden)
        self.linear2 = nn.Linear(actor_hidden, 128)
        self.linear3 = nn.Linear(128, actor_hidden)

        # Define a dense nn for the actor
        self.actor = nn.Linear(actor_hidden, self.environment.n_actions)
        # Another one for the critic
        self.critic = nn.Linear(critic_hidden, 1)

    def forward(self, x):

        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)
        x = F.relu(x)

        return x

    def get_action_probs(self, x):
        # Action probability (basically the policy)
        x = self(x)
        action_probs = F.softmax(self.actor(x), dim=-1)
        return action_probs

    def evaluate_actions(self, x):
        # Evaluation of the action quality by the critic
        x = self(x)
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return action_probs, state_values

    def loss(self, advantages, action_gain):
        # Computing the loss that is given by the advantages as seen in the TD-error trick
        value_loss = advantages.pow(2).mean()
        loss = value_loss - action_gain
        return loss

    def update_agent(self, rewards, states, actions, optimizer):
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
        a = Variable(torch.LongTensor(actions).view(-1, 1))
        chosen_action_log_probs = action_log_probs.gather(1, a)
        action_gain = (chosen_action_log_probs * advantages).mean()
        total_loss = self.loss(advantages=advantages, action_gain=action_gain)

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        optimizer.step()

    def train(self):
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

            # only reflecting/training on episodes where a failure occurred. No training
            if len(rewards) < 200:
                # signal in perfect games. 
                # Reflect phase (taking the critic feedback into account)
                if i % 50 == 0:
                    print("Training : ", i)
                    print("         * Score : ", len(rewards))

                self.update_agent(rewards=rewards,
                                  states=states,
                                  actions=actions,
                                  optimizer=optimizer)

            else:
                if i % 50 == 0:
                    print("Not Training : ", i)
                    print("         * Score : ", len(rewards))

            scores.append(len(rewards))

        return np.array(scores)
