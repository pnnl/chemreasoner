"""Script to test the parallel calls to the llm."""
import argparse
import pickle
import sys
import time

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("src")
from search.reward.llm_reward import LLMRewardFunction  # noqa: E402
from search.state.reasoner_state import ReasonerState  # noqa: E402
from search.methods.tree_search.beam_search import BeamSearchTree  # noqa: E402
from search.policy.reasoner_policy import ReasonerPolicy  # noqa: E402

adsorbates = ["CO", "H2O"]
template = (
    "Generate a list of top-5 {catalyst_label} "
    f"for the adsorption of {' and '.join(adsorbates)}."
    "{include_statement}{exclude_statement}"
    "Provide scientific explanations for each of the catalysts. "
    "Finally, return a python list named final_answer which contains the top-5 catalysts. "
    "{candidate_list_statement}"
    r"\n\nTake a deep breath and let's think step-by-step. Remember, you need to return a python list named final_answer!"
)
starting_state = ReasonerState(
    template=template,
    reward_template=None,
    ads_symbols=adsorbates,
    ads_preferences=[1, -1],
    num_answers=5,
    prediction_model="llama-2",
    reward_model="llama-2",
    debug=True,
)
policy = ReasonerPolicy(
    catalyst_label_types=["", "monometallic ", "bimetallic ", "trimetallic "],
    try_oxides=False,
)


def llm_function(prompts, system_prompts):
    if len(prompts) > 1:
        print("\n*" * 5)
    elif len(prompts) == 1:
        print("\n------" * 5)
    else:
        print("\n~~~~~~~" * 5)
    return ["final_answer =['Pt','Cu','Ag']" for p, s in zip(prompts, system_prompts)]


reward_function = LLMRewardFunction(
    llm_function=llm_function,
)


search = BeamSearchTree(starting_state, policy, reward_function, 4, 3)

for i in range(3):
    search.simulation_policy()


n_batch = 3

states = [starting_state.copy() for _ in range(n_batch)]

# print(reward_function(states))
