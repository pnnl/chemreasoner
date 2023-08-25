"""Base class for reward functions."""
from abc import ABCMeta, abstractmethod


class BaseReward(metaclass=ABCMeta):
    """A base class for reward functions."""

    @abstractmethod
    def __call__(self, s: object):
        """Return the reward for a given state."""
        ...
