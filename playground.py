"""
Playground script for tests
"""

import sys
sys.path.append('C:/Users/lilia/OneDrive/Documents/GitHub/RL/src/env')
from environment import Environment
sys.path.append('C:/Users/lilia/OneDrive/Documents/GitHub/RL/src/policies')
from neural_policy import NeuralNetPolicy
sys.path.append('C:/Users/lilia/OneDrive/Documents/GitHub/RL/src/optimizers')
from pgo import PGO
sys.path.append('C:/Users/lilia/OneDrive/Documents/GitHub/RL/src/viz')
import visualization as viz
import matplotlib.pyplot as plt

environment = Environment(Environment.CART_POL_V1, seed=0)

policy = NeuralNetPolicy(environment=environment, hidden_layer_size=256, learning_rate=0.003)
pgo = PGO(policy=policy, horizon=500, gamma=0.99)
pgo_scores = pgo.train(max_trajectory=500, printEvery=50)

viz.score_visualisation(pgo_scores, f"PGO Scores on {environment.id}")

plt.show()
