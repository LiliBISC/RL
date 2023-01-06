from abc import ABC, abstractmethod


class AbstractPolicy(ABC):
    """
    Abstract class of any policy

    This class is meant for programming purposes and can't be instantiated
    """

    def __init__(self, args):
        # TODO: what do we need in every policies ?
        pass

    @abstractmethod
    def any_common_method(self, args):
        # TODO: what method is to be shared between all policies ?
        pass
