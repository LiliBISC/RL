"""
Playground script for tests
"""

from src.env.environment import Environment
from src.policies.neural_policy import NeuralNetPolicy
from src.optimizers.pgo import PGO
import src.viz.visualization as viz
import matplotlib.pyplot as plt

environment = Environment(Environment.CART_POL_V1)

policy = NeuralNetPolicy(environment=environment, hidden_size=256, learning_rate=0.003)
pgo = PGO(environment=environment, policy=policy, horizon=500, gamma=0.99)
pgo_scores = pgo.train(max_trajectory=500, printEvery=50)

viz.score_visualisation(pgo_scores, f"PGO Scores on {environment.id}")

plt.show()
