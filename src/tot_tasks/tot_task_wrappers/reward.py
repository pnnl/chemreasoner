"""A reward wrapper for tot tasks."""
import functools
import sys

sys.path.insert(0, "src/tot_tasks/tree-of-thought-llm")
import run  # noqa: E402
from models import gpt  # noqa: E402

run.gpt = functools.partial(gpt, model="gpt-3.5-turbo", temperature=0.7)


def extract_reward(state):
    """Get the reward stored in the state."""
    return state.value


def value_ys(task_obj, task_state, ys, n_evaluate_sample, cache_value=True):
    """Calculate the values for a given set of ys."""
    return run.get_values(
        task_obj, task_state.x, ys, n_evaluate_sample, cache_value=True
    )
