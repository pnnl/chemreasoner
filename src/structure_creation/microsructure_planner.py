"""Moduel for microstructure creation from LLM answers."""

import logging
import sys
import time

sys.path.append("src")
from search.state.reasoner_state import ReasonerState

logging.getLogger().setLevel(logging.INFO)


class MicrostructurePlanner:
    """A class to handle LLM-driven microstructure creation."""

    def __init__(self, llm_function=callable):
        """Init self."""
        self.llm_function = llm_function

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
        "Answer": "(3) Zinc Oxide: This catalyst is good because...",
        "catalyst_name": "Zinc Oxide",
        "bulk_structure": "mp-123",
        "millers": (1, 2, 3),
        "cell_shift": 3,
        "site_placement": (3.2, 4.5, 6.7),
        "atoms_objeect_id": "{db_id}_{structure_id}",
    }
}
