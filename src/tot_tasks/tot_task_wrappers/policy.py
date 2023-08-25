"""Wapper funcitons to translate the ToT examples to MCR."""
import functools
import sys

import numpy as np

sys.path.insert(0, "src/tot_tasks/tree-of-thought-llm")
import tasks.game24 as game24  # noqa: E402
import models  # noqa: E402

sys.path.insert(0, "src/tot_tasks/tot_task_wrappers")
import reward  # noqa: E402


def task_action(task_state, task_obj):
    """Propose new y from a state."""
    propose_prompt = task_obj.propose_prompt_wrap(task_state.x, task_state.y)
    proposals = models.gpt(propose_prompt, n=1, model="gpt-3.5-turbo", stop=None)[
        0
    ].split("\n")
    new_state = task_state.copy()
    ys = [task_state.y + p + "\n" for p in proposals]
    ys_values = reward.value_ys(task_obj, task_state, ys, 10)
    max_idx = np.argmax(ys_values)
    y = ys[max_idx]
    new_state.y = y
    new_state.value = ys_values[max_idx]
    new_state.depth = task_state.depth + 1
    return new_state


class _SelectorAction:
    def __init__(self, idx):
        self.idx = idx

    def __call__(self, eval_state):
        return eval_state.TaskState(x=eval_state.x, y=eval_state.y, next_ys=[])


class TaskPolicy:
    """Policy for tot tasks."""

    def __init__(self, num_actions, task_obj, depth_limit=4):
        """Initialize the selector actions."""
        self.task_obj = task_obj
        self.actions = []
        for i in range(num_actions):
            self.actions.append(functools.partial(task_action, task_obj=task_obj))

    def get_actions(self, task_state):
        """Return the available actions and priors."""
        if task_state.depth >= 4:
            return (self.actions, np.zeros_like(self.actions))
        else:
            return (self.actions, np.ones_like(self.actions) / len(self.actions))

    def early_stopping(self, state):
        """Whether to stop the search early. Always False for this policy."""
        return self.task_obj.test_output(state.task_idx, state.y)["r"]


if __name__ == "__main__":
    game24.DATA_PATH = "src/tot_tasks/tree-of-thought-llm/data"
    print(game24.Game24Task())
    models.gpt("test", n=1, model="gpt-3.5-turbo", stop=None)[0].split("\n")
