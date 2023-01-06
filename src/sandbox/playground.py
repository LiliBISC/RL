"""
Playground script for tests
"""

import gym
from src.policies.neural_policy import NeuralNetPolicy
from src.optimizers.pgo import PGO

env = gym.make('CartPole-v1')

policy = NeuralNetPolicy(env=env, hidden_size=256, learning_rate=0.003)
pgo = PGO(env=env, policy=policy, horizon=500, gamma=0.99)
pgo.train(max_trajectory=500, printEvery=50)
