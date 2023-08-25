"""Functions to run mcts."""
import argparse
import pickle
import sys
import time

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("src")
from llm import automate_prompts  # noqa: E402
from search.reward import llm_reward  # noqa: E402
from search.methods.tree_search import mcts, beam_search  # noqa: E402

# from search.methods.sampling import single_shot, multi_shot  # noqa: E402


def single_shot(starting_state, directory, fname):
    """Save a single_shot_query."""
    start_time = time.time()
    starting_state.query()
    end_time = time.time()
    saving_data = vars(starting_state)
    saving_data["node_rewards"] = llm_reward.llm_adsorption_energy_reward(
        starting_state
    )
    saving_data["start_time"] = start_time
    saving_data["end_time"] = end_time
    with open(directory / ("single_shot_" + fname), "wb") as f:
        pickle.dump(saving_data, f)


def multi_shot(starting_state, directory: Path, fname, num_trials=10):
    """Save a single_shot_query."""
    for j in range(10):
        starting_state = starting_state.copy()
        starting_state.query()
        saving_data = vars(starting_state)
        saving_data["node_rewards"] = llm_reward.llm_adsorption_energy_reward(
            starting_state
        )
        with open(directory / (f"multi_shot_{j}_" + fname), "wb") as f:
            pickle.dump(saving_data, f)


def main(args):
    """Run the search on desired inputs."""
    if "oc" in Path(args.input).stem:
        adsorbates = np.loadtxt(args.input, dtype=str)
        fname = "oc_db"

        prompt_iterator = enumerate(adsorbates)
        state_policy_generator = automate_prompts.get_initial_state_oc

    elif "biofuels" in Path(args.input).stem:
        df = pd.read_csv(args.input)
        fname = Path(args.input).stem

        prompt_iterator = df.iterrows()
        state_policy_generator = automate_prompts.get_initial_state_biofuels

    for i, prompt in prompt_iterator:
        print(prompt)
        starting_state, policy = state_policy_generator(
            prompt, "gpt-3.5-turbo", "gpt-3.5-turbo"
        )
        if "single_shot" in args.methods:
            single_shot(starting_state.copy(), Path(args.savedir), f"{fname}_{i}.pkl")

        if "multi_shot" in args.methods:
            multi_shot(
                starting_state.copy(),
                Path(args.savedir),
                f"{fname}_{i}.pkl",
                num_trials=10,
            )

        if "mcts" in args.methods:
            # Do single shot and multi shot querying.
            single_shot(starting_state, Path(args.savedir), f"{fname}_{i}.pkl")

            # Make a new starting state for the tree search
            reward = llm_reward.llm_adsorption_energy_reward
            tree = mcts.MonteCarloTree(
                data=starting_state.copy(),
                policy=policy,
                reward_fn=reward,
                tradeoff=15,
                discount_factor=0.9,
            )
            tree.start_timer()
            max_steps = 250
            for i in range(max_steps):
                print(f"---- {i} ----")
                tree.step_save(Path(args.savedir) / f"mcts_{fname}_{i}.pkl")

        if "beam_search" in args.methods:
            reward = llm_reward.llm_adsorption_energy_reward
            tree = beam_search.BeamSearchTree(
                data=starting_state,
                policy=policy,
                reward_fn=reward,
                num_generate=12,
                num_keep=6,
            )
            tree.start_timer()
            num_levels = 6
            for i in range(num_levels):
                print(f"---- {i} ----")
                tree.step_save(Path(args.savedir) / f"beam_search_{fname}_{i}.pkl")
        else:
            raise NotImplementedError(
                f"Search method {args.method} is not implemented."
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("savedir", type=str)
    parser.add_argument("input", type=str)
    parser.add_argument("--methods", type=str, nargs="+", default=["single_shot"])

    args = parser.parse_args()
    main(args)
