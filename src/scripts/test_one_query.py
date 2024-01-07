"""Test a single query."""
import argparse
import sys

from pathlib import Path

import pandas as pd

sys.path.append("src")
from datasets import reasoner_data_loader  # noqa:E402
from search.policy import coherent_policy, reasoner_policy  # noqa:E402
from search.reward import simulation_reward, reaction_reward, llm_reward  # noqa:E402
from search.methods.tree_search.beam_search import BeamSearchTree  # noqa:E402
from llm.azure_open_ai_interface import run_azure_openai_prompts  # noqa:E402

df = pd.read_csv("data/input_data/dataset.csv", index_col=False)
first_row = df.iloc[0]
# TODO: Check if output file exists and is nonzero
starting_state = reasoner_data_loader.get_state(
    first_row["dataset"], first_row["query"]
)

policy = coherent_policy.CoherentPolicy(run_azure_openai_prompts)

starting_state.catalyst_label = " metallic catalysts"
starting_state.priors_template = coherent_policy.priors_template
starting_state.relation_to_candidate_list = (
    starting_state.relation_to_candidate_list
    if starting_state.relation_to_candidate_list is not None
    else "similar to"
)
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
search = BeamSearchTree(starting_state, policy, reward_fn, 4, 3)

for i in range(5):
    try:
        data = search.step_return()
    except Exception as err:
        # TODO: dump the data and be able to load from where it left off
        ...
        raise err

    print("=" * 20 + " " + str(i) + " " + "=" * 20)

# TODO: Save final data in a json tree in separate file from other queries. One file per query
# test_Xquery_y.json
