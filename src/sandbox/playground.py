"""
Playground script for tests
"""

import gym
from src.policies.neural_policy import NeuralNetPolicy
from src.optimizers.pgo import PGO
import src.viz.visualization as viz
import matplotlib.pyplot as plt

dataset = 'CartPole-v1'

env = gym.make(dataset)

policy = NeuralNetPolicy(env=env, hidden_size=256, learning_rate=0.003)
pgo = PGO(env=env, policy=policy, horizon=500, gamma=0.99)
pgo_scores = pgo.train(max_trajectory=500, printEvery=50)

viz.score_visualisation(pgo_scores, f"PGO Scores on {dataset}")

plt.show()
