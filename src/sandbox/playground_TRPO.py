from src.env.environment import Environment
from src.optimizers.trpo import TRPO
import src.viz.visualization as viz
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


def classic():
    environment = Environment(Environment.CART_POL_V1, seed=0, max_duration=1000)
    trpo = TRPO(environment=environment, learning_rate=0.003, horizon=500, actor_hidden=64, critic_hidden=64)

    trpo_score = trpo.train(num_rollouts=5)

    viz.score_visualisation(np.array(trpo_score), f"TRPO Scores on {environment.id}")

    plt.show()


def kl_graph(kl_list, duration=400):
    scores = np.zeros((duration, len(kl_list)))

    environment = Environment(Environment.CART_POL_V1, seed=0, max_duration=1000)

    for i, kl_value in enumerate(kl_list):
        print(f"[*] TRPO with KL parameter {kl_value}")
        trpo = TRPO(environment=environment,
                    learning_rate=0.003,
                    horizon=duration,
                    actor_hidden=64,
                    critic_hidden=64,
                    max_d_kl=kl_value)

        scores[:, i] = trpo.train(num_rollouts=2)

        plt.show()

    df = pd.DataFrame(columns=kl_list, data=scores)

    viz.score_visualisation(df, f"TRPO Scores on {environment.id}", show_variance=True)

    plt.show()

def env_trpo(seed_lst, duration):
    scores = np.zeros((duration, len(seed_lst)))

    for i, seed in enumerate(seed_lst):
        environment = Environment(Environment.CART_POL_V1, seed=seed, max_duration=1000)
        print(f"[*] PGO with environment {environment}")
        trpo = TRPO(environment=environment,
                    learning_rate=0.003,
                    horizon=duration,
                    actor_hidden=64,
                    critic_hidden=64,
                    max_d_kl=1e-2)

        scores[:, i] = trpo.train(num_rollouts=2)

    df = pd.DataFrame(columns=seed_lst, data=scores)

    viz.score_visualisation(df, f"TRPO Scores on different environments")

    plt.show()


env_trpo([0,1,2,3,4], duration = 400)
