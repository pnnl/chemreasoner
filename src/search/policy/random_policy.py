"""Class for the random policies."""
import numpy as np


class RandomUniformPolicy:
    """Policy that perturbs atoms, adds atoms, removes atoms."""

    def __init__(self, name_action_pairs: list[tuple[str, callable]]):
        """Save thegiven actions."""
        self.action_names, self.actions = zip(*name_action_pairs)

    def get_action(self, state):
        """Return an action and prior_logit for given state."""
        return (
            self.actions[np.random.randint(low=0, high=len(self.actions))],
            1 / len(self.actions),
        )

    def get_actions(self, state):
        """Return a actions and prior_logits for given state."""
        return (
            self.actions,
            np.array([1 / len(self.actions)] * len(self.actions)),
        )

    @staticmethod
    def early_stopping(*args):
        """Whether to stop the search early. Always False for this policy."""
        return False


class WeightedPolicy:
    """Policy that perturbs atoms, adds atoms, removes atoms."""

    def __init__(
        self, name_action_pairs: list[tuple[str, callable]], weights=list[float]
    ):
        """Save the given actions and weights."""
        self.action_names, self.actions = zip(*name_action_pairs)
        self.weights = weights

    def get_action(self, state):
        """Return an action and prior_logit for given state."""
        random_index = np.random.coice(range(len(self.actions)), p=self.weights)
        return (
            self.actions[random_index],
            self.weights[random_index],
        )

    def get_actions(self, state):
        """Return a actions and prior_logits for given state."""
        return (
            self.actions,
            self.weights,
        )

    @staticmethod
    def early_stopping(*args):
        """Whether to stop the search early. Always False for this policy."""
        return False
