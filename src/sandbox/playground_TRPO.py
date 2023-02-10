from src.env.environment import Environment
from src.policies.neural_policy import NeuralNetPolicy
from src.optimizers.trpo import TRPO
import src.viz.visualization as viz
import matplotlib.pyplot as plt
import numpy as np

environment = Environment(Environment.CART_POL_V1, seed=0, max_duration=1000)
trpo = TRPO(environment=environment, learning_rate=0.003, horizon=500, actor_hidden=64, critic_hidden=64)

trpo_score = trpo.train(num_rollouts=5)

viz.score_visualisation(np.array(trpo_score), f"TRPO Scores on {environment.id}")

plt.show()