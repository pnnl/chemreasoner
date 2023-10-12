"""A base class for policies."""
from abc import ABCMeta, abstractmethod, Callable, Optional

import numpy as np


class BasePolicy(metaclass=ABCMeta):
    """A base class for policies."""

    @abstractmethod
    def get_actions(
        self, states: list[object]
    ) -> tuple[list[Callable[object, object]], list[np.array]]:
        """Return the actions along with their priors."""
        ...

    @abstractmethod
    def early_stopping(self):
        """Whether or not to stop the search early."""
        ...
