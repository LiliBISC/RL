"""
Playground script for tests
"""

from src.env.environment import Environment
from src.policies.neural_policy import NeuralNetPolicy
from src.optimizers.A2C import A2C
import src.viz.visualization as viz
import matplotlib.pyplot as plt

environment = Environment(Environment.CART_POL_V1, seed=0)

a2c = A2C(environment = environment, learning_rate = 0.003, horizon = 500)
a2c_scores = a2c.train_test()

viz.score_visualisation(a2c_scores, f"A2C Scores on {environment.id}")

plt.show()