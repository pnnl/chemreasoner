"""Moduel for microstructure creation from LLM answers for the OCP format."""

import logging
import sys
import time

from typing import Optional

sys.path.append("src")
from search.state.reasoner_state import ReasonerState

logging.getLogger().setLevel(logging.INFO)


class OCPMicrostructurePlanner:
    """A class to handle LLM-driven microstructure creation using the OCP format."""

    default_retries = {
        "symbols": 2,
        "bulk": 0,
        "millers": 0,
        "cell_shift": 0,
        "adsorbate_site": 0,
        "__default__": 1,
    }

    def __init__(self, llm_function=callable):
        """Init self."""
        self.llm_function = llm_function

    def process_prompt(
        self,
        prompt_info: list[str],
        prompt_type,
        prompt_creation_function: callable[str, str | None],
        prompt_parsing_function: callable[str, str | None],
        system_prompt_function: callable[str, str | None],
        retries: Optional[int] = None,
        **llm_function_kwargs,
    ):
        """A script to handle the creation, running, parsing of a prompt."""
        return_value = False
        if isinstance(prompt_info, list):
            return_value = True
            prompt_info = [prompt_info]
        if retries is None:
            if prompt_type in self.default_retries:
                retries = self.default_retries[prompt_type]
            else:
                retries = self.default_retries["__default__"]

        attempts = 0
        answers = [None] * len(prompt_info)
        while attempts <= retries and any([ans is None for ans in answers]):
            # Get the prompts
            prompt_idx = []
            prompts = []
            system_prompts = []
            for i, p_info in prompt_info:
                if answers[i] is None:  # If answer hasn't been found, yet
                    proposed_prompt = prompt_creation_function(p_info)
                    if proposed_prompt is not None:  # If prompt creation is successful
                        prompt_idx.append(i)
                        prompts.append(prompt_creation_function(p_info))
                        system_prompts.append(system_prompt_function(p_info))

            # Run non-None prompts through the LLM to get raw answers
            raw_answers = self.llm_function(
                prompts, system_prompts, **llm_function_kwargs
            )
            for p_idx, raw_answer in zip(prompt_idx, raw_answers):
                # Attempt to parse out the answer
                answers[p_idx] = prompt_parsing_function(raw_answer)

        if return_value:
            return answers[0]
        else:
            return answers

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
        self, symbols: list[list[str]], states: list[ReasonerState]
    ):
        """Run the generation prompts for the given states where the reward is None.

        Updates the given "slab_syms" list in-place.
        """
        start = time.time()
        prompts = []
        system_prompts = []
        prompts_idx = []
        for i, s in enumerate(states):
            if symbols[i] is None:
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
                    symbols[state_idx] = s.process_catalyst_symbols(answers[i])

                except Exception as err:
                    logging.warning(f"Failed to parse answer with error: {str(err)}.")
            end = time.time()
            logging.info(
                f"TIMING: Slab symbols parsing finished in reward function {end-start}"
            )

    def run_bulk_prompt(self, slab_symbols):
        """Run the bulk prompt for the given slab symbols."""
        ...

    def run_millers_prompt(self, bulks):
        """Run the bulk prompt for the given slab symbols."""
        ...

    def compute_cell_shifts(self, bulk, millers):
        """Compute the possible cell shifts for the given bulk and millers."""
        ...

    def run_cell_shift_prompt(self, slabs):
        """Run the bulk prompt for the given slab symbols."""
        ...

    def compute_site_placements(self, slab):
        """Compute the possible site placements for the given slab."""
        ...

    def run_site_plancement_prompt(self, bulks):
        """Run the bulk prompt for the given slab symbols."""
        ...


example_data_structure = {
    {
        "llm_answer": "(3) Zinc Oxide: This catalyst is good because...",
        "catalyst_name": "Zinc Oxide",
        "symbols": ["Zn", "O"],
        "bulk_structure": "mp-123",
        "millers": (1, 2, 3),
        "cell_shift": 3,
        "site_placement": (3.2, 4.5, 6.7),
        "atoms_object_id": "{db_id}_{structure_id}",
    }
}
