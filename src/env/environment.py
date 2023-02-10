import gym
import random
import torch
import numpy as np


class Environment(object):
    """
    This is a class to instantiate an environment in a practical way and enumerate used environments
    """

    CART_POL_V1 = 'CartPole-v1'

    def __init__(
            self,
            env_id: str,
            seed: int = None,
            max_duration: int = None,
    ):

        if env_id not in gym.envs.registry.keys():
            raise Exception(
                f"{env_id} is not a registered environment in gym. \nAvailable environments: \n{list(gym.envs.registry.keys())}")

        self.id = env_id
        self.seed = seed
        self.max_duration = max_duration

        self.env = gym.make(env_id, max_episode_steps=max_duration, autoreset=False)
        self.n_observations = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        if seed is not None:
            self.seed_everything()

    def seed_everything(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def reset(self):
        return self.env.reset(seed=self.seed)

    def step(self, action):
        return self.env.step(action)

    def observation_space_shape(self):
        """Get the shape of the observations"""
        return self.env.observation_space.shape

    def action_space_shape(self):
        """Get the shape of the actions"""
        return self.env.action_space.shape
