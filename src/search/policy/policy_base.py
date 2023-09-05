"""A base class for policies."""
from abc import ABCMeta, abstractmethod, Callable, Optional

import numpy as np


class BasePolicy(metaclass=ABCMeta):
    """A base class for policies."""

    @abstractmethod
    def get_actions(
        self, state: object, trial: Optional[bool] = False
    ) -> tuple[list[Callable[object, object]], np.array]:
        """Return the actions along with their priors."""
        ...
