import gym


class Environment(object):
    """
    This is a class to instantiate an environment in a practical way and enumerate used environments
    """

    CART_POL_V1 = 'CartPole-v1'

    def __init__(
            self,
            env_id: str,
            seed: int = None,
    ):
        if env_id not in gym.envs.registry.keys():
            raise Exception(
                f"{env_id} is not a registered environment in gym. \nAvailable environments: \n{list(gym.envs.registry.keys())}")

        self.id = env_id
        self.seed = seed

        self.env = gym.make(env_id)
        self.n_observations = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

    def reset(self):
        return self.env.reset(seed=self.seed)

    def step(self, action):
        return self.env.step(action)


