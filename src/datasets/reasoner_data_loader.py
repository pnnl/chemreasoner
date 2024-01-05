"""A class to load reasoner states, policies, and reward functions for state."""
import sys

from pathlib import Path

import pandas as pd

_default_dataset = pd.read_csv(Path("data", "llm_answers.csv"))
sys.path.append("src")
from llm.automate_prompts import (  # noqa:E402
    get_initial_state_open_catalyst,
    get_initial_state_bio_fuels,
    get_initial_state_rwgs,
    get_initial_state_ethanol,
    get_initial_state_methanol,
)


def get_state(dataset, prompt, chain_of_thought=True):
    """Get the ReasonerState, Policy, reward_function for the given index."""

    if dataset == "OpenCatalyst":
        return get_initial_state_open_catalyst(
            prompt,
            prediction_model=None,
            reward_model=None,
            chain_of_thought=chain_of_thought,
        )
    elif dataset == "BioFuels":
        return get_initial_state_bio_fuels(
            prompt,
            prediction_model=None,
            reward_model=None,
            chain_of_thought=chain_of_thought,
        )
    elif dataset == "RWGS":
        return get_initial_state_rwgs(
            prompt,
            prediction_model=None,
            reward_model=None,
            chain_of_thought=chain_of_thought,
        )
    elif dataset == "CO2ToEthanol":
        return get_initial_state_ethanol(
            prompt,
            prediction_model=None,
            reward_model=None,
            chain_of_thought=chain_of_thought,
        )
    elif dataset == "CO2ToMethanol":
        return get_initial_state_methanol(
            prompt,
            prediction_model=None,
            reward_model=None,
            chain_of_thought=chain_of_thought,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset}.")


if __name__ == "__main__":
    df = pd.read_csv("data/input_data/dataset.csv")
    new_data = {"dataset": [], "query": [], "cot": [], "no_cot": []}
    print(len(df))
    for i, data in df.iterrows():
        cot_state = get_state(data["dataset"], data["query"], chain_of_thought=True)
        no_cot_state = get_state(data["dataset"], data["query"], chain_of_thought=False)
        new_data["dataset"].append(data["dataset"])
        new_data["query"].append(data["query"])
        new_data["cot"].append(cot_state.generation_prompt)
        new_data["no_cot"].append(no_cot_state.generation_prompt)
    pd.DataFrame(new_data).to_csv("test_df.csv", index=False)
