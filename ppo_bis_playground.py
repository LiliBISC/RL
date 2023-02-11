from src.env.environment import Environment
from src.policies.neural_policy import NeuralNetPolicy
from src.optimizers.ppo_bis import PPO
from src.optimizers.ppo_a2c import PPO_A2C
from src.optimizers.ppo_comprehensive import PPO as PPO_comprehensive
import src.viz.visualization as viz
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/lilia/OneDrive/Documents/GitHub/RL/src/viz')
import src.viz.visualization as viz
import numpy as np
import pandas as pd
import ipdb

def clipping_graph(clipping_values, duration=300):
    scores = np.zeros((duration, len(clipping_values)))

    environment = Environment(Environment.CART_POL_V1, seed=0, max_duration=1000)

    for i, clipping_coef in enumerate(clipping_values):
        print(f"[*] PPO with clipping {clipping_coef}")
        ppo = PPO_comprehensive(
            environment=environment,
            clipping=clipping_coef,
            learning_rate=0.003,
            horizon=duration,
            coef_entropy=0.1,
            coef_value=0.1,
        )

        scores[:, i] = ppo.train(num_rollouts=2)

        plt.show()

    df = pd.DataFrame(columns=clipping_values, data=scores)

    viz.score_visualisation(df, f"PPO Scores on {environment.id}", show_variance=True)

    plt.show()

def values_graph(coef_values, duration=300):
    scores = np.zeros((duration, len(coef_values)))

    environment = Environment(Environment.CART_POL_V1, seed=0, max_duration=1000)

    for i, coef_value in enumerate(coef_values):
        print(f"[*] PPO with coef_entropy {coef_value}")
        ppo = PPO_comprehensive(
            environment=environment,
            clipping=0.2,
            learning_rate=0.003,
            horizon=duration,
            coef_entropy=coef_value,
            coef_value=0.1,
        )

        scores[:, i] = ppo.train(num_rollouts=2)

        plt.show()

    df = pd.DataFrame(columns=coef_values, data=scores)

    viz.score_visualisation(df, f"PPO Scores on {environment.id}", show_variance=True)

    plt.show()

if __name__=="__main__":
    environment = Environment(Environment.CART_POL_V1, seed=0, max_duration=1000)
    # clipping_graph([0.01,0.2,0.4])
    values_graph([0.0,0.2,0.4])
    # environment = gym.make('CartPole-v1')
    # ipdb.set_trace()
    # ppo = PPO(
    #     environment = environment, 
    #     learning_rate=0.003, 
    #     horizon=200, 
    #     actor_hidden = 64, 
    #     critic_hidden= 64,

    # )
    # ppo_a2C = PPO_A2C(
    #     environment=environment,
    #     learning_rate=0.003,
    #     horizon=2000,
    #     coef_entropy=0.1,
    #     coef_values=0.1
    # )
    # ppo_comphrensive_usual_clipping = PPO_comprehensive(
    #     environment = environment, 
    #     clipping=0.2,
    #     learning_rate=0.003, 
    #     horizon=200,
    #     coef_entropy=0.1,
    #     coef_value=0.1,
    # )
    # # ppo_scores= ppo_a2C.train_test()
    # ppo_scores = ppo_comphrensive_usual_clipping.train(num_rollouts=2)
    # # ppo_scores = ppo.train()
    # viz.score_visualisation(np.array(ppo_scores))
    # plt.show()
