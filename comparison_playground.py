from src.env.environment import Environment
from src.policies.neural_policy import NeuralNetPolicy
from src.optimizers.trpo import TRPO
from src.optimizers.actor_critic import A2C
from src.optimizers.pgo import PGO
from src.optimizers.ppo_comprehensive import PPO as PPO_comprehensive
import src.viz.visualization as viz
import matplotlib.pyplot as plt
import sys

import src.viz.visualization as viz
import numpy as np
import pandas as pd
import ipdb


learning_rate = 0.003
horizon = 300
if __name__ == "__main__":
    environment = Environment(Environment.CART_POL_V1, seed=0,max_duration=1000)
    policy = NeuralNetPolicy(
        environment=environment, hidden_layer_size=256, learning_rate=learning_rate
    )
    pgo = PGO(policy=policy, horizon=horizon, gamma=0.99)
    a2c = A2C(environment=environment, learning_rate=learning_rate, horizon=horizon)
    trpo = TRPO(
        environment=environment,
        learning_rate=learning_rate,
        horizon=horizon,
        actor_hidden=64,
        critic_hidden=64,
        max_d_kl=0.01,
    )
    ppo = PPO_comprehensive(
        environment=environment,
        learning_rate=learning_rate,
        horizon=horizon,
        coef_entropy=0.1,
        coef_value=0.1,
        max_d_kl=0.0,
    )
    
    policies = [pgo, a2c, trpo, ppo]
    labels = ["pgo","a2c","trpo","ppo"]
    scores = np.zeros((horizon, len(labels)))
    try:
        scores[:,0] = pgo.train(max_trajectory=horizon, printEvery=100)
        scores[:,1] = a2c.train()
        scores[:,2] = trpo.train(num_rollouts=2)
        scores[:,3] = ppo.train(num_rollouts=2) 
    except:
        ipdb.set_trace()

    df_scores = pd.DataFrame(scores, columns=[f"{element.upper()}_score" for element in labels])
    viz.score_visualisation(df_scores, show_variance=False)
    ipdb.set_trace()
