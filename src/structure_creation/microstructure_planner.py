"""Moduel for microstructure creation from LLM answers for the OCP format."""

import logging
import os
import pickle
import sys
import time

from ast import literal_eval
from typing import Optional
import crystal_toolkit

import mp_api
from pymatgen.core.surface import SlabGenerator
from pymatgen.ext.matproj import MPRester

from ocdata.core import Slab


sys.path.append("src")
from llm.utils import process_prompt
from search.state.reasoner_state import ReasonerState
from structure_creation.digital_twin import SlabDigitalTwin

logging.getLogger().setLevel(logging.INFO)

MP_API_KEY = os.environ["MP_API_KEY"]


prompts = {
    "bulk_structure": {
        "prompt": (
            "$ROOT_PROMPT = {root_prompt}\n\n"
            "Consider the following list of materials. \n"
            "Return the index of the material that would be best suited for answering the $ROOT_PROMPT.\n\n"
            "{bulks_summaries}"
            "\nReturn your answer as a python list called 'final_answer' of your top two choices, using the indices given in the () above. Let's think step-by-step and provide justifications for your answers."
        ),
        "system_prompt": (
            "You are an AI assistant that has knowledge about materials science and can make accurate recommendations about bulk material structures based on their crystal structure and composition. You will consider factors such as catalytic performance and synthesizability in your analysis."
        ),
    },
    "millers": {
        "prompt": (
            "$ANSWER = {answer}\n\n"
            "$ROOT_PROMPT = {root_prompt}\n\n"
            "Consider the material {material}. "
            "Return a list of miller indices that would answer the $ROOT_PROMPT. You miller indices should be consistent with the information in $ANSWER\n\n"
            "\nReturn your answer as a python list called 'final_answer' of your top two miller indices. Let's think step-by-step and provide justifications for your answers."
        ),
        "system_prompt": (
            "You are an AI assistant that has knowledge about materials science and catalysis and can make accurate recommendations about surface composition based on surface chemistry. You will consider factors such as catalytic performance and binding sites."
        ),
    },
    "cell_shift": {
        "prompt": "$ROOT_PROMPT = {root_prompt}\n\nConsider the material {material} with miller index {millers}. Return the index of the following surfaces which has the best configuration for accomplishing the $ROOT_PROMPT. You should target surfaces with binding sites that are consisten with the $ANSWER given above.\n\n{cell_shift_summaries}\n\nReturn your answer as a python list called 'final_answer' of your top three choices, using the indices given in the () above. Let's think step-by-step and provide justifications for your answers.",
        "system_prompt": (
            "You are an AI assistant that has knowledge about materials science and catalysis and can make accurate recommendations about miller indices based on surface chemistry. You will consider factors such as catalytic performance and binding sites."
        ),
    },
    "site_placement": {
        "prompt": (
            "$ROOT_PROMPT = {root_prompt}\n\n"
            "Consider the material {material} with miller index {millers}. "
            "Return the index of the atomic environment of the best adsorbate placement site.\n\n"
            "{atomic_environments}"
            "\nReturn your answer as a python list called 'final_answer' of your top two choices, using the indices given in the () above. Let's think step-by-step and provide justifications for your answers."
        ),
        "system_prompt": (
            "You are an AI assistant that has knowledge about materials science and catalysis and can make accurate recommendations about adsorbate binding sites based on surface chemistry. You will consider factors such as catalytic performance and binding sites."
        ),
    },
}


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
        prompt_type: str,
        prompt_creation_function: callable,
        prompt_parsing_function: callable,
        system_prompt_function: callable,
        retries: Optional[int] = None,
        **llm_function_kwargs,
    ):
        """A script to handle the creation, running, parsing of a prompt."""
        if retries is None:
            if prompt_type in self.default_retries:
                retries = self.default_retries[prompt_type]
            else:
                retries = self.default_retries["__default__"]

        return process_prompt(
            self.llm_function,
            prompt_info,
            prompt_creation_function,
            prompt_parsing_function,
            system_prompt_function,
            retries=retries,
            **llm_function_kwargs,
        )

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


class BulkSelector:
    prompt_templates = prompts["bulk_structure"]
    _example_bulk_answer = """In order to convert CO2 to methanol, a catalyst is required that can promote the reduction of CO2. ZnO is widely known to be a good catalyst for this process due to its ability to adsorb and activate CO2. ZnO2, however, is not typically used for this purpose. So, we can exclude the ZnO2 materials from consideration.\n\nNext, we need to consider the impact of the crystal structure on the catalytic performance. Here are some general rules:\n\n1. The cubic Fm-3m space group ZnO has a high symmetry and densely packed atoms, which might not provide enough surface area and active sites for CO2 conversion. \n\n2. The hexagonal P6_3mc space group ZnO has lower symmetry and loosely packed atoms, which could provide more surface area and active sites for CO2 conversion.\n\n3. The cubic F-43m space group ZnO has a high symmetry and densely packed atoms, similar to the cubic Fm-3m space group. \n\nBased on this information, the hexagonal P6_3mc space group ZnO would be expected to perform better due to its larger surface area and more available active sites. However, the cubic F-43m space group ZnO might also perform well due to its high stability and good synthesizability, despite its high symmetry.\n\nTherefore, the top two choices would be:\n\n`final_answer = [2, 4]` \n\nThis answer is based on the known properties of these materials and their crystal structures. However, the actual performance could vary and should be verified through experiments."""

    @staticmethod
    def fetch_materials(symbols: list[str]) -> list["MPDataDoc"]:
        """Fetch mateirals in the materials project database with given symbols."""
        with MPRester(MP_API_KEY) as mpr:
            docs = mpr.summary.search(elements=symbols)

        return [d for d in docs if all([str(elem) in symbols for elem in d.elements])]

    @staticmethod
    def convert_to_dict(docs: list["MPDataDoc"]) -> list[dict]:
        """Conver MPDataDoc objects to dicts for prompt creation."""
        dicts = []
        for doc in docs:
            dicts.append(
                {
                    "mp-id": str(docs[0].material_id),
                    "formation-energy-per-atom": doc.formation_energy_per_atom,
                    "energy-above-hull": doc.formation_energy_per_atom,
                    "is-stable": doc.is_stable,
                    "formula-pretty": doc.formula_pretty,
                    "n-sites": doc.nsites,
                    "theoretical": doc.theoretical,
                    "efermi": doc.efermi,
                    "symmetry-cystal-system": doc.symmetry.crystal_system.value.lower(),
                    "symmetry-symbols": doc.symmetry.symbol,
                }
            )
        return dicts

    @staticmethod
    def filter_materials(docs: list["MPDataDoc"]) -> list["MPDataDoc"]:
        """Filter materials lsited in dictionaries."""
        return [d for d in docs if not d.theoretical]

    @staticmethod
    def create_prompt(docs: list["MPDataDoc"], state) -> str:
        """Create a prompt for the given dictionaries."""
        bulks_summaries = ""
        for i, doc in enumerate(docs):
            bulks_summaries += f"({i}) {doc.formula_pretty} in the {doc.symmetry.crystal_system.value.lower()} {doc.symmetry.symbol} space group.\n"

        return fstr(prompts["prompt"], {"bulks_summaries": bulks_summaries})

    @staticmethod
    def parse_response_list(response: str) -> list[int]:
        """Create a prompt for the given dictionaries."""
        for line in response.split("\n"):
            if "final_answer" in line:
                list_start = line.find("[")
                list_end = line.find("]")
                answer_list = literal_eval(line[list_start : list_end + 1])
        return answer_list


def compute_subsurface_distribution(slab: Slab):
    """Compute distribution of atoms under the surface of slab."""
    ...


def fstr(fstring_text, vals):
    """Evaluate the provided fstring_text."""
    ret_val = eval(f'f"{fstring_text}"', vals)
    return ret_val


example_data_structure = [
    {
        "llm_answer": "(3) Zinc Oxide: This catalyst is good because...",
        "catalyst_name": "Zinc Oxide",
        "symbols": ["Zn", "O"],
        "bulk_structure": "mp-2133",
        "millers": (1, 2, 3),
        "cell_shift": 3,
        "site_placement": (3.2, 4.5, 6.7),
        "atoms_object_id": "{db_id}_{structure_id}",
    }
]

if __name__ == "__main__":
    dt = SlabDigitalTwin(computational_params={"answer": "Zinc Oxide"})
    dt.set_symbols(["Zn", "O"])
    bulks = dt.get_bulks()
    with open("bulks_tmp.pkl", "wb") as f:
        pickle.dump(bulks, f)
    dt.set_bulk([bulks[0]])
    dt.set_millers([(1, 0, 0)])
    print(dt.get_surfaces())
