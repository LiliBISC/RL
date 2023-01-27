"""
Playground script for tests
"""

import sys
sys.path.append('C:/Users/lilia/OneDrive/Documents/GitHub/RL/src/env')
from environment import Environment
sys.path.append('C:/Users/lilia/OneDrive/Documents/GitHub/RL/src/policies')
from neural_policy import NeuralNetPolicy
sys.path.append('C:/Users/lilia/OneDrive/Documents/GitHub/RL/src/optimizers')
from A2C import A2C
sys.path.append('C:/Users/lilia/OneDrive/Documents/GitHub/RL/src/viz')
import visualization as viz
import matplotlib.pyplot as plt

environment = Environment(Environment.CART_POL_V1, seed=0)

a2c = A2C(environment = environment)
a2c_scores = a2c.train_test(N_GAMES=500)

viz.score_visualisation(a2c_scores, f"A2C Scores on {environment.id}")

plt.show()
