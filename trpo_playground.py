from src.env.environment import Environment
from src.policies.neural_policy import NeuralNetPolicy
from src.optimizers.TRPO import TRPO
import src.viz.visualization as viz
import matplotlib.pyplot as plt
import gym
import sys
sys.path.append('C:/Users/lilia/OneDrive/Documents/GitHub/RL/src/viz')
import src.viz.visualization as viz
import numpy as np
import ipdb


if __name__=="__main__":
    environment = Environment(Environment.CART_POL_V1, seed=0)

    # policy = NeuralNetPolicy(environment=environment, hidden_layer_size=256, learning_rate=0.003)
    # optimizer = PPO(
    #     policy=policy,
    #     horizon=500,
    #     gamma=0.99,
    #     clipping=0.2,
    #     coef_entropy=1.2,
    #     coef_value=1,
    #     lmbda=0.8
    # )

    # scores = optimizer.train(max_trajectory=50, printEvery=50)
    environment = gym.make('CartPole-v1')
    trpo = TRPO(environment = environment, learning_rate=0.003, horizon=70, actor_hidden = 32, critic_hidden= 32)

    trpo_score = trpo.train(num_rollouts=5)

    viz.score_visualisation(np.array(trpo_score))
    plt.show()
    ipdb.set_trace()
