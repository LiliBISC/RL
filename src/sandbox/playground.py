from src.env.environment import Environment
from src.optimizers.actor_critic import A2C
from src.policies.neural_policy import NeuralNetPolicy
from src.optimizers.pgo import PGO
import src.viz.visualization as viz
import matplotlib.pyplot as plt

environment = Environment(Environment.CART_POL_V1, seed=0, max_duration=1000)

a2c = A2C(environment=environment, learning_rate=3e-3, horizon=400)
a2c_scores = a2c.train()


policy = NeuralNetPolicy(environment=environment, hidden_layer_size=256, learning_rate=2e-3)
pgo = PGO(policy=policy, horizon=300, gamma=0.99)
pgo_scores = pgo.train(max_trajectory=400, printEvery=50)


ppo = PPO(
            environment=environment,
            clipping=0.2,
            learning_rate=0.003,
            horizon=300,
            coef_entropy=coef_value,
            coef_value=0.1,
        )