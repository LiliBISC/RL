"""
Playground script for tests
"""

from src.env.environment import Environment
from src.optimizers.actor_critic import A2C
import src.viz.visualization as viz
import matplotlib.pyplot as plt

environment = Environment(Environment.CART_POL_V1, seed=0, max_duration=1000)

a2c = A2C(environment=environment, learning_rate=3e-3, horizon=400)
a2c_scores = a2c.train()

viz.score_visualisation(a2c_scores, f"A2C Scores on {environment.id}")

plt.show()