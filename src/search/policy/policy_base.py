"""A base class for policies."""
from abc import ABCMeta, abstractmethod
from collections.abc import Callable

import numpy as np


class BasePolicy(metaclass=ABCMeta):
    """A base class for policies."""

    @abstractmethod
    def get_actions(
        self, states: list[object]
    ) -> tuple[list[Callable], list[np.array]]:
        """Return the actions along with their priors."""
        ...
