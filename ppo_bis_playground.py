from src.env.environment import Environment
from src.policies.neural_policy import NeuralNetPolicy
from src.optimizers.ppo_bis import PPO
from src.optimizers.ppo_a2c import PPO_A2C
from src.optimizers.ppo_comprehensive import PPO as PPO_comprehensive
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
    environment = gym.make('CartPole-v1')
    # ipdb.set_trace()
    ppo = PPO(
        environment = environment, 
        learning_rate=0.003, 
        horizon=200, 
        actor_hidden = 64, 
        critic_hidden= 64,

    )
    ppo_a2C = PPO_A2C(
        environment=environment,
        learning_rate=0.003,
        horizon=2000,
        coef_entropy=0.1,
        coef_values=0.1
    )
    ppo_comphrensive = PPO_comprehensive(
        environment = environment, 
        learning_rate=0.003, 
        horizon=120,
        coef_entropy=0.1,
        coef_value=0.1,
    )
    # ppo_scores= ppo_a2C.train_test()
    ppo_scores = ppo_comphrensive.train()
    # ppo_scores = ppo.train()
    viz.score_visualisation(np.array(ppo_scores))
    plt.show()
