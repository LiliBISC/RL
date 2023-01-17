"""
Playground for the PPO scripts
"""

from src.env.environment import Environment
from src.policies.neural_policy import NeuralNetPolicy
from src.optimizers.ppo import PPO
import src.viz.visualization as viz
import matplotlib.pyplot as plt

environment = Environment(Environment.CART_POL_V1, seed=0)

policy = NeuralNetPolicy(environment=environment, hidden_layer_size=256, learning_rate=0.003)
optimizer = PPO(
    policy=policy,
    horizon=500,
    gamma=0.99,
    clipping=0.2,
    coef_entropy=1.2,
    coef_value=1,
    lmbda=0.8
)

scores = optimizer.train(max_trajectory=500, printEvery=50)

viz.score_visualisation(scores, f"PPO Scores on {environment.id}")

plt.show()
