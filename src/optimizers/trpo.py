from abstract_optimizer import AbstractOptimizer


class TRPO(AbstractOptimizer):
    """
    Implementation of Trust Region Policy Optimization
    """

    def __init__(self, args):
        super().__init__(args)

    def any_common_method(self, args):
        raise Exception("Not implemented")