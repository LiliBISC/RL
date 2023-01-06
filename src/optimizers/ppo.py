from abstract_optimizer import AbstractOptimizer


class PPO(AbstractOptimizer):
    """
    Implementation of Proximal Policy Optimization
    """

    def __init__(self, args):
        super().__init__(args)

    def any_common_method(self, args):
        raise Exception("Not implemented")
