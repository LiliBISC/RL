import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.optimizers.actor_critic import A2C
from src.optimizers.trpo import TRPO
from src.optimizers.ppo_comprehensive import PPO
from src.optimizers.pgo import PGO
from src.env.environment import Environment
from src.policies.neural_policy import NeuralNetPolicy
from src.viz.visualization import score_visualisation

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def env_trpo(seed_lst, duration):
    plt.figure(figsize=(20, 20))
    for i, seed in enumerate(seed_lst):
        environment = Environment(Environment.CART_POL_V1, seed=seed, max_duration=1000)
        print(f"[*] All models with environment {Environment.CART_POL_V1} and seed {seed}")
        ppo = PPO(
            environment=environment,
            learning_rate=3e-3,
            horizon=duration,
            coef_entropy=0.1,
            coef_value=0.1,
            max_d_kl=1e-2,
        )
        trpo = TRPO(environment=environment,
                    learning_rate=3e-3,
                    horizon=duration,
                    actor_hidden=64,
                    critic_hidden=64,
                    max_d_kl=1e-2)
        a2c = A2C(environment=environment,
                  learning_rate=3e-3,
                  horizon=duration)
        policy = NeuralNetPolicy(environment=environment, hidden_layer_size=256, learning_rate=3e-3)
        pgo = PGO(policy=policy, horizon=500, gamma=0.99)

        pgo_scores = pgo.train(max_trajectory=duration, printEvery=50)
        trpo_scores = trpo.train(num_rollouts=2)
        a2c_scores = a2c.train()
        ppo_scores = ppo.train(num_rollouts=2)

        scores = np.concatenate((pgo_scores.reshape(-1,1), a2c_scores.reshape(-1,1)), axis=1)
        scores = np.concatenate((scores, np.array(trpo_scores).reshape(-1,1)), axis=1)
        scores = np.concatenate((scores, np.array(ppo_scores).reshape(-1,1)), axis=1)

        df = pd.DataFrame(columns=['PGO', 'A2C', 'TRPO', 'PPO'], data=scores)

        plt.subplot(421 + i)
        score_visualisation(df, f"All scores on environment {Environment.CART_POL_V1} and seed {seed}", figure=False, show_variance=False)
    plt.show()

env_trpo([0, 12, 22, 13, 2, 9, 50, 40], duration = 250)