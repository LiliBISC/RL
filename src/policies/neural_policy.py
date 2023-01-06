from abstract_policy import AbstractPolicy


class NeuralNetPolicy(AbstractPolicy):
    """
    Abstract class of any policy

    This class is meant for programming purposes and can't be instantiated
    """

    def __init__(self, args):
        super().__init__(args)

    def any_common_method(self, args):
        raise Exception("Not implemented yet")
