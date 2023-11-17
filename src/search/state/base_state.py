"""Base class for a search state."""
from abc import ABCMeta, abstractmethod, Callable, Optional


class BaseState(metaclass=ABCMeta):
    """Base class for a search state."""

    @abstractmethod
    @classmethod
    @staticmethod
    def from_dict(data: dict) -> "BaseState":
        """Return the state constructed from the given dictionary."""
        ...

    @abstractmethod
    def copy(self) -> "BaseState":
        """Return a copy of self."""
        ...

    @abstractmethod
    def return_next(self) -> "BaseState":
        """Return the successor state of self."""
        ...
