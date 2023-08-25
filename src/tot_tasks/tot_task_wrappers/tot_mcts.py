"""Functions to run with tot tasks."""
import argparse
import pickle
import sys

from pathlib import Path

sys.path.insert(0, "src/tot_tasks/tree-of-thought-llm")
import tasks.game24 as game24  # noqa: E402

sys.path.insert(0, "src/tot_tasks/tot_task_wrappers")
import reward  # noqa: E402
import policy  # noqa: E402
import state  # noqa: E402

sys.path.insert(0, "src/search/tree_search")
from tree import SearchTree, process_tree_data  # noqa: E402


def step_save_with_backoff(tree, fname):
    """Take a step with the tree and save the tree, backing off on failure."""
    tree.simulation_policy()
    best_state, reward = tree.get_best_state(reward=True)

    saving_tree = process_tree_data(tree)
    with open(fname, "wb") as f:
        pickle.dump(saving_tree, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("savedir", type=str)
    args = parser.parse_args()

    game24.DATA_PATH = "src/tot_tasks/tree-of-thought-llm/data"
    task_obj = game24.Game24Task()
    policy_fn = policy.TaskPolicy(5, task_obj)
    reward_fn = reward.extract_reward
    for starting_idx in range(150):
        starting_state = state.TaskState(
            task_obj.get_input(starting_idx), task_idx=starting_idx
        )
        starting_state.value = 0.0

        tree = SearchTree(
            data=starting_state,
            policy=policy_fn,
            reward_fn=reward_fn,
            tradeoff=5,
            discount_factor=1.0,
        )
        max_steps = 100
        for i in range(max_steps):
            print(f"---- {i} ----")
            step_save_with_backoff(
                tree, Path(args.savedir) / f"game24-{starting_idx}.pkl"
            )
