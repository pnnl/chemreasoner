"""Test a single query."""
import argparse
import json
import os
import sys
import time

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("src")
from datasets import reasoner_data_loader  # noqa:E402
from search.policy import coherent_policy, reasoner_policy  # noqa:E402
from search.reward import simulation_reward, reaction_reward, llm_reward  # noqa:E402
from search.state.reasoner_state import ReasonerState  # noqa:E402
from search.methods.tree_search.beam_search import BeamSearchTree  # noqa:E402
from llm.azure_open_ai_interface import run_azure_openai_prompts  # noqa:E402


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


df = pd.read_csv("data/input_data/dataset.csv", index_col=False)
first_row = df.iloc[0]

starting_state = reasoner_data_loader.get_state(
    first_row["dataset"], first_row["query"]
)

policy = coherent_policy.CoherentPolicy(run_azure_openai_prompts)
reward_fn = simulation_reward.StructureReward(
    **{
        "llm_function": run_azure_openai_prompts,
        "model": "gemnet",
        "traj_dir": Path("data/output/henry_debug_1/5/24"),
        "device": "cpu",
        "ads_tag": 2,
        "num_adslab_samples": 1,
        "num_slab_samples": 2,
        "steps": 2,
    },
)

if (
    Path("test_tree_timing.json").exists()
    and os.stat("test_tree_timing.json").st_size != 0
):
    with open("test_tree.json", "r") as f:
        tree_data = json.load(f)
        search = BeamSearchTree.from_data(
            tree_data, policy, reward_fn, node_constructor=ReasonerState.from_dict
        )
else:
    starting_state.catalyst_label = " metallic catalysts"
    starting_state.priors_template = coherent_policy.priors_template
    starting_state.relation_to_candidate_list = (
        starting_state.relation_to_candidate_list
        if starting_state.relation_to_candidate_list is not None
        else "similar to"
    )

    search = BeamSearchTree(
        starting_state, policy, lambda list_x: [2] * len(list_x), 5, 5
    )

start_time = time.time()
timing_data = [start_time]
for i in range(5):
    try:
        data = search.step_return()
        end_time = time.time()
        timing_data.append(end_time - timing_data[-1])
        with open("test_tree_timing.json", "w") as f:
            data.update(
                {"total_time": end_time - start_time, "step_times": timing_data}
            )
            json.dump(data, f, cls=NpEncoder)
    except Exception as err:
        raise err

    print("=" * 20 + " " + str(i) + " " + "=" * 20)
