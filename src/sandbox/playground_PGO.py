"""
Playground script for tests
"""

from src.env.environment import Environment
from src.policies.neural_policy import NeuralNetPolicy
from src.optimizers.pgo import PGO
import src.viz.visualization as viz
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


def classic_pgo():
    environment = Environment(Environment.CART_POL_V1, seed=0, max_duration=1000)

    policy = NeuralNetPolicy(environment=environment, hidden_layer_size=256, learning_rate=2e-3)
    pgo = PGO(policy=policy, horizon=500, gamma=0.99)
    pgo_scores = pgo.train(max_trajectory=400, printEvery=50)

    viz.score_visualisation(pgo_scores, f"PGO Scores on {environment.id}")

    plt.show()


def learning_rate_pgo(learning_rates, duration):
    scores = np.zeros((duration, len(learning_rates)))

    environment = Environment(Environment.CART_POL_V1, seed=0, max_duration=1000)

    for i, learning_rate in enumerate(learning_rates):
        print(f"[*] PGO with learning rate {learning_rate}")
        environment.reset()
        policy = NeuralNetPolicy(environment=environment, hidden_layer_size=256, learning_rate=learning_rate)
        pgo = PGO(policy=policy, horizon=500, gamma=0.99)
        pgo_scores = pgo.train(max_trajectory=duration, printEvery=50)

        scores[:, i] = pgo_scores

    df = pd.DataFrame(columns=learning_rates, data=scores)

    viz.score_visualisation(df, f"PGO Scores on {environment.id}")

    plt.show()


def env_pgo(env_lst, duration):
    scores = np.zeros((duration, len(env_lst)))

    for i, e in enumerate(env_lst):
        environment = Environment(e, seed=0, max_duration=1000)
        print(f"[*] PGO with environment {e}")
        environment.reset()
        policy = NeuralNetPolicy(environment=environment, hidden_layer_size=256, learning_rate=3e-3)
        pgo = PGO(policy=policy, horizon=500, gamma=0.99)
        pgo_scores = pgo.train(max_trajectory=duration, printEvery=50)

        scores[:, i] = pgo_scores

    df = pd.DataFrame(columns=env_lst, data=scores)

    viz.score_visualisation(df, f"PGO Scores on different environments")

    plt.show()


env_pgo([Environment.CART_POL_V1, Environment.AIR_RAID_V5], duration = 400)