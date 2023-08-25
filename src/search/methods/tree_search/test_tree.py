"""Classes to set up a tree search test."""
import numpy as np


class _TestState:
    def __init__(self, string=""):
        self.string = string

    def _return_next(self):
        return _TestState(self.string)

    def __str__(self):
        return self.string

    def __repr__(self):
        return self.string


class _TestNumberAdder:
    def __init__(self, number):
        self.number = str(number)

    def __call__(self, test_state):
        new_state = test_state._return_next()
        new_state.string += self.number
        return new_state


class _TestPolicy:
    def __init__(self, num_actions):
        self.actions = []
        for i in range(num_actions):
            self.actions.append(_TestNumberAdder(i))

    def get_actions(self, test_state):
        """Get actions and priors for a test state."""
        self.priors = np.ones_like(self.actions)

        if test_state.string != "":
            for i, a in enumerate(self.actions):
                if a.number in test_state.string[-3:]:
                    self.priors[i] = 0

        return (self.actions, self.priors / np.sum(self.priors))


def _test_reward(test_state):
    r = 0
    for char in test_state.string:
        r += int(char)
    return r
