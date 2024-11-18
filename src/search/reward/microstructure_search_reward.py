"""Module for reward function by calculation of adsorption energies in simulation."""

import logging
import os
import sys
import time

from pathlib import Path

import numpy as np
import pandas as pd
from ase.data import chemical_symbols
from pymatgen.ext.matproj import MPRester

sys.path.append("src")
from llm.ase_interface import StructureGenerationError
from search.state.reasoner_state import ReasonerState
from search.reward.base_reward import BaseReward
from search.methods.tree_search.microstructure_tree_search import (
    run_microstructure_search,
)

logging.getLogger().setLevel(logging.INFO)

print("finished imports")

MP_API_KEY = os.environ["MP_API_KEY"]


class StructureReward(BaseReward):
    """Calculate the reward for answers based on adsorption simulations."""

    def __init__(
        self,
        llm_function: callable,
        config,
        microstructure_results_dir: Path,
        penalty_value: float = -10,
        max_attempts: int = 3,
    ):
        """Select the class of nnp for the reward function."""
        self.llm_function = llm_function
        self.penalty_value = penalty_value
        self.max_attempts = max_attempts
        self.microstructure_results_dir = (
            microstructure_results_dir
            if isinstance(microstructure_results_dir, Path)
            else Path(microstructure_results_dir)
        )
        self.config = config

    def run_generation_prompts(
        self, slab_syms: list[list[str]], states: list[ReasonerState]
    ):
        """Run the generation prompts for the given states where the reward is None."""
        start = time.time()
        prompts = []
        system_prompts = []
        for i, s in enumerate(states):
            if slab_syms[i] is None:
                prompts.append(s.generation_prompt)
                system_prompts.append(s.generation_system_prompt)

        if len(prompts) > 0:
            generation_results = self.llm_function(
                prompts, system_prompts, **{"temperature": 0.7, "top_p": 0.95}
            )
            loop_counter = 0
            for i, s in enumerate(states):
                if slab_syms[i] is None:
                    try:
                        s.process_generation(generation_results[loop_counter])
                    except Exception:
                        logging.info("failed to process generation answer.")
                        pass
                    loop_counter += 1
            end = time.time()
            logging.info(
                f"TIMING: Candidate generation finished in reward function {end-start}"
            )

    def run_slab_sym_prompts(
        self, slab_syms: list[list[str]], states: list[ReasonerState]
    ):
        """Run the generation prompts for the given states where the reward is None.

        Updates the given "slab_syms" list in-place.
        """
        start = time.time()
        prompts = []
        system_prompts = []
        prompts_idx = []
        for i, s in enumerate(states):
            if slab_syms[i] is None:
                try:
                    prompts.append(s.catalyst_symbols_prompt)
                    system_prompts.append(None)
                    prompts_idx.append(i)
                except Exception as err:
                    logging.warning(
                        f"Failed to generate prompts with error: {str(err)}. "
                        "Skipping this prompt."
                    )
                    if len(prompts) > len(system_prompts):
                        prompts.pop()
        if len(prompts) > 0:
            answers = self.llm_function(
                prompts, system_prompts, **{"temperature": 0.01, "top_p": 0.01}
            )
            logging.info(answers)

            for i, p in enumerate(prompts):
                state_idx = prompts_idx[i]
                s = states[state_idx]
                try:
                    slab_syms[state_idx] = s.process_catalyst_symbols(answers[i])

                except Exception as err:
                    logging.warning(f"Failed to parse answer with error: {str(err)}.")
            end = time.time()
            logging.info(
                f"TIMING: Slab symbols parsing finished in reward function {end-start}"
            )

    def __call__(
        self,
        states: list[ReasonerState],
        num_attempts: int = 3,
        primary_reward: bool = True,
        query_name="co_to_methanol",
    ):
        """Run the microstructure planner for the predicted catalysts."""
        slab_syms = [None] * len(states)
        attempts = 0
        start = time.time()
        while any([s is None for s in slab_syms]) and attempts < self.max_attempts:
            if primary_reward:
                self.run_generation_prompts(slab_syms, states)

            self.run_slab_sym_prompts(slab_syms, states)

            attempts += 1
        end = time.time()
        logging.info(
            f"TIMING: Candidate/symbol generation finished in reward function {end-start}"
        )
        state_rewards = []
        for i, s in enumerate(states):
            node_rewards_data = {}
            candidates_list = s.candidates
            if slab_syms[i] is None:
                logging.warning(
                    f"Unable to parse the answer:\n\n {s.answer}."
                    "\n\nInto catalyst symbols. "
                    "Returning the penalty value for that answer."
                )
                state_rewards.append(self.penalty_value)
            else:
                retry_logs = []
                for j, candidate, symbols in zip(
                    range(len(candidates_list)), candidates_list, slab_syms[i]
                ):
                    try:
                        if any(
                            [
                                s not in chemical_symbols
                                or chemical_symbols.index(s) > 82
                                or chemical_symbols.index(s) < 11
                                for s in symbols
                            ]
                        ):
                            raise StructureGenerationError(
                                f"Cannot create bulk with slab_syms {symbols}."
                            )
                        results_dir = (
                            self.microstructure_results_dir
                            / f"{'_'.join([s.lower() for s in symbols])}_{query_name}"
                        )
                        rewards_csv_path = results_dir / "reward_values.csv"
                        if not rewards_csv_path.exists():
                            if len(get_available_bulks(symbols)) == 0:
                                raise StructureGenerationError(
                                    f"No available bulks for the symbols {symbols}."
                                )
                            else:
                                results_dir.mkdir(parents=True, exist_ok=True)
                                attempts = 0
                                while attempts < self.max_attempts:
                                    try:
                                        attempts += 1
                                        dataframe = run_microstructure_search(
                                            self.config, symbols, results_dir
                                        )
                                        node_rewards_data[candidate] = (
                                            self.process_dataframe(dataframe)
                                        )
                                    except Exception as err:
                                        if attempts == self.max_attempts:
                                            node_rewards_data[candidate] = (
                                                self.penalty_value
                                            )
                                        logging.warning(
                                            f"Microstructure Search failed with error {err}."
                                        )
                                        retry_logs.append(
                                            f"Microstructure Search failed with error {err}."
                                        )
                        else:
                            logging.info(
                                f"Reading results for {candidate} from file {rewards_csv_path}."
                            )
                            dataframe = pd.read_csv(rewards_csv_path)
                            node_rewards_data[candidate] = self.process_dataframe(
                                dataframe
                            )

                    except StructureGenerationError as err:
                        logging.warning(err)
                        node_rewards_data[candidate] = self.penalty_value
                # Logging here to save any info from micro search in state s

                final_reward = self.aggregate_rewards(node_rewards_data)
                state_rewards.append(final_reward)
        return state_rewards

    def process_dataframe(self, dataframe):
        """Process a reward from the given dataframe."""
        return np.max(dataframe["reward"])

    def aggregate_rewards(self, node_rewards):
        """Turn node rewards into a reward value."""
        print(node_rewards)
        return np.mean(list(node_rewards.values()))


def get_available_bulks(syms):
    """Get the bulks available for the given syms."""
    with MPRester(MP_API_KEY) as mpr:
        docs = mpr.summary.search(elements=syms)
    # Filter for materials with only the specified elements
    docs = [d for d in docs if all([str(elem) in syms for elem in d.elements])]
    return docs
