"""Functions to run mcts."""
import argparse
import pickle
import sys
import time

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.append("src")
from llm import automate_prompts  # noqa: E402
from search.reward import simulation_reward, llm_reward  # noqa: E402
from search.policy.coherent_policy import CoherentPolicy, ReasonerPolicy  # noqa: E402
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


def main(args, policy_string):
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
            prompt,
            "gpt-3.5-turbo",
            "gpt-3.5-turbo",
            simulation_reward=args.reward_function == "simulation-reward",
        )
        if args.policy == "coherent-policy":
            policy = CoherentPolicy.from_reasoner_policy(policy)
        if "single_shot" in args.search_methods:
            single_shot(starting_state.copy(), Path(args.savedir), f"{fname}_{i}.pkl")

        if "multi_shot" in args.search_methods:
            multi_shot(
                starting_state.copy(),
                Path(args.savedir),
                f"{fname}_{i}.pkl",
                num_trials=10,
            )

        if "mcts" in args.search_methods:
            # Do single shot and multi shot querying.

            if args.reward_function == "llm-reward":
                reward = llm_reward.llm_adsorption_energy_reward
            elif args.reward_function == "simulation-reward":
                reward = simulation_reward.StructureReward(
                    num_adslab_samples=16, num_slab_samples=16, device="cuda:0"
                )

            tree = mcts.MonteCarloTree(
                data=starting_state.copy(),
                policy=policy,
                reward_fn=reward,
                tradeoff=15,
                discount_factor=0.9,
            )
            tree.start_timer()
            max_steps = 300
            for j in range(max_steps):
                print(f"---- {j} ----")
                tree.step_save(
                    Path(args.savedir) / f"mcts_{policy_string}_{fname}_{i}.pkl"
                )

        if "beam_search" in args.search_methods:
            if args.reward_function == "llm-reward":
                reward = llm_reward.llm_adsorption_energy_reward
            elif args.reward_function == "simulation-reward":
                reward = simulation_reward.StructureReward(
                    num_adslab_samples=16,
                    num_slab_samples=16,
                    device="cuda:0",
                    model="gemnet",
                    traj_dir=Path("data/output_data/trajectories/pipeline_test"),
                )
            tree = beam_search.BeamSearchTree(
                data=starting_state,
                policy=policy,
                reward_fn=reward,
                num_generate=12,
                num_keep=6,
            )
            tree.start_timer()
            num_levels = 7
            for j in range(num_levels):
                print(f"---- {j} ----")
                tree.step_save(
                    Path(args.savedir) / f"beam_search_{policy_string}_{fname}_{i}.pkl"
                )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str)
    parser.add_argument("--savedir", type=str)
    parser.add_argument("--llm", type=str)
    parser.add_argument("--search_method", type=str)
    parser.add_argument("--policy", type=str)
    parser.add_argument("--reward", type=str)

    args = parser.parse_args()

    savedir = Path(args.savedir).mkdir(exist_ok=True, parents=True)

    Path("data", "output_data", "demo", "oc", "test").mkdir(parents=True, exist_ok=True)

    main(args, policy_string=args.policy)

    