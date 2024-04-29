"""Moduel for microstructure creation from LLM answers for the OCP format."""

import logging
import os
import pickle
import sys
import time

from ast import literal_eval
from typing import Optional

from ase.data import chemical_symbols

import crystal_toolkit

import mp_api
from pymatgen.core.surface import SlabGenerator
from pymatgen.ext.matproj import MPRester

import numpy as np
from ocdata.core import Slab

sys.path.append("src")
from llm.utils import process_prompt
from search.state.reasoner_state import ReasonerState
from structure_creation.digital_twin import SlabDigitalTwin

logging.getLogger().setLevel(logging.INFO)

MP_API_KEY = os.environ["MP_API_KEY"]


prompts = {
    "bulk": {
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
            "$ROOT_PROMPT = {root_prompt}\n\n"
            "Consider the material {material}. \n"
            "Return a list of miller indices that would answer the $ROOT_PROMPT. You miller indices should be consistent with the information in $ANSWER\n\n"
            "\nReturn your answer as a python list called 'final_answer' of your top two miller indices. Let's think step-by-step and provide justifications for your answers."
        ),
        "system_prompt": (
            "You are an AI assistant that has knowledge about materials science and catalysis and can make accurate recommendations about surface composition based on surface chemistry. You will consider factors such as catalytic performance and binding sites."
        ),
    },
    "surface": {
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

    def evaluate_states(self, states: list[ReasonerState]):
        """Run the microstructure planner for the given set of reasoner states."""
        self.digital_twins = []
        self.states = states

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

    @staticmethod
    def literal_parse_response_list(response: str) -> list[int]:
        """Create a prompt for the given dictionaries."""
        for line in response.split("\n"):
            if "final_answer" in line:
                list_start = line.find("[")
                list_end = line.find("]")
                answer_list = literal_eval(line[list_start : list_end + 1])
        return answer_list

    @staticmethod
    def create_bulk_prompt(twin_state: tuple[SlabDigitalTwin, ReasonerState]):
        """Create the prompt for bulks."""
        twin, state = twin_state
        bulks = twin_state.get_bulks()
        bulks_summaries = ""
        for i, doc in enumerate(bulks):
            bulks_summaries += f"({i}) {doc.formula_pretty} in the {doc.symmetry.crystal_system.value.lower()} {doc.symmetry.symbol} space group.\n"

        prompt_values = {
            "bulks_summaries": bulks_summaries,
            "root_prompt": state.root_prompt,
        }
        prompt = fstr(prompts["bulk"], prompt_values)
        twin.update_info("bulk", {"prompt": prompt})
        return

    def create_bulk_system_prompt(
        self, twin_state: tuple[SlabDigitalTwin, ReasonerState]
    ):
        """Create millers system prompt."""
        twin, state = twin_state
        prompt = prompts["bulk"]["system_prompt"]
        twin.update_info("bulk", {"system_prompt": prompt})
        return prompt

    @staticmethod
    def parse_bulk_answer(
        self, answer, twin_state: tuple[SlabDigitalTwin, ReasonerState]
    ):
        """Parse the bulk_prompt_response."""
        # TODO: Track the behavior here
        twin, state = twin_state
        answer = answer["answer"]
        usage = answer["usage"]
        info = {"answer": answer, "usage": usage}

        answer_list = self.parse_response_list(answer)
        twin.update_info("bulk", info)
        return answer_list

    def select_bulks(self, digital_twins: SlabDigitalTwin, states: ReasonerState):
        """Run the bulk prompt for the given slab symbols."""
        twin_states = [(d, s) for d, s in zip(digital_twins, states)]
        bulks_idxs = self.process_prompt(
            self.llm_function,
            twin_states,
            self.create_bulk_prompt,
            self.parse_bulk_answer,
            self.create_bulk_system_prompt,
            # TODO: LLM function kwargs
        )
        length_twins = len(digital_twins)
        for i in range(length_twins):
            ans = bulks_idxs[i]
            digital_twin = digital_twins[i]
            selected_bulks = [digital_twin.get_bulks()[j] for j in ans]
            digital_twins += digital_twin.set_bulk(selected_bulks)

    def create_millers_prompt(self, twin_state: tuple[SlabDigitalTwin, ReasonerState]):
        """Create a prompt for the miller index."""
        twin, state = twin_state
        doc = twin.computational_objects["bulks"]
        values = {
            "root_prompt": state.root_prompt,
            # "answer": state.answer,
            "material": f"{doc.formula_pretty} in the {doc.symmetry.crystal_system.value.lower()} {doc.symmetry.symbol} space group.\n",
        }
        prompt = fstr(prompts["millers"]["prompt"], values)
        twin.update_info("millers", {"prompt": prompt})
        return prompt

    def create_millers_system_prompt(
        self, twin_state: tuple[SlabDigitalTwin, ReasonerState]
    ):
        """Create millers system prompt."""
        twin, state = twin_state
        prompt = prompts["millers"]["system_prompt"]
        twin.update_info("millers", {"system_prompt": prompt})
        return prompt

    def run_millers_prompt(self, digital_twins: SlabDigitalTwin, states: ReasonerState):
        """Run the bulk prompt for the given slab symbols."""
        twin_states = [(d, s) for d, s in zip(digital_twins, states)]
        millers_choices = self.process_prompt(
            self.llm_function,
            twin_states,
            self.create_millers_prompt,
            self.parse_millers_answer,
            self.create_millers_system_prompt,
            # TODO: LLM function kwargs
        )
        length_twins = len(digital_twins)
        for i in range(length_twins):
            millers = millers_choices[i]
            digital_twin = digital_twins[i]
            digital_twins += digital_twin.set_millers(millers)

    @staticmethod
    def create_site_placement_prompt(twin_state: tuple[SlabDigitalTwin, ReasonerState]):
        """Create the prompt for site_placement."""
        twin, state = twin_state
        site_placements = twin.get_site_placements()
        if len(site_placements) > 0:
            site_placements_summaries = []
            for site in site_placements:
                site_placements_summaries.append(describe_site_placement(site))
        else:
            return None
        doc = twin.computational_objects["bulk"]
        values = {
            "root_prompt": state.root_prompt,
            "material": f"{doc.formula_pretty} in the {doc.symmetry.crystal_system.value.lower()} {doc.symmetry.symbol} space group.\n",
            "millers": f"{twin.computational_params['millers']}",
        }
        prompt = fstr(prompts["site_placement"]["prompt"], values)
        twin.update_info("site_placement", {"prompt": prompt})
        return prompt

    @staticmethod
    def create_site_placement_system_prompt(
        twin_state: tuple[SlabDigitalTwin, ReasonerState]
    ):
        """Create the prompt for site_placement."""
        twin, state = twin_state
        prompt = prompts["site_placement"]["system_prompt"]
        twin.update_info("site_placement", {"system_prompt": prompt})
        return prompt

    def run_site_placement_prompt(
        self, digital_twins: SlabDigitalTwin, states: ReasonerState
    ):
        """Run the bulk prompt for the given slab symbols."""
        twin_states = [(d, s) for d, s in zip(digital_twins, states)]
        millers_choices = self.process_prompt(
            self.llm_function,
            twin_states,
            self.create_millers_prompt,
            self.parse_millers_answer,
            self.create_site_placement_system_prompt,
            # TODO: LLM function kwargs
        )
        length_twins = len(digital_twins)
        for i in range(length_twins):
            millers = millers_choices[i]
            digital_twin = digital_twins[i]
            digital_twins += digital_twin.set_millers(millers)


def get_neighbors_site(surface: Slab, site: tuple, cutoff=2.5):
    """Get the neighboring atoms of the given site."""
    print(site)
    site = np.array(site)
    diffs = surface.atoms.get_positions() - site
    distances = np.linalg.norm(diffs, axis=1)
    return surface.atoms.get_atomic_numbers()[distances <= cutoff]


def describe_neighbors_site(neighbors):
    """Linguistically describe the given neigbhors."""
    if len(neighbors) == 0:
        return "Far from any atoms on the surface."
    elif len(neighbors) == 1:
        return f"Near a single {chemical_symbols[neighbors[0]]}."
    else:
        counts = {}
        for z in np.unique(neighbors):
            counts[chemical_symbols[z]] = len(
                [z_prime for z_prime in neighbors if z == z_prime]
            )
        return (
            "Near "
            + ", ".join([f"{count} {elem}" for elem, count in counts.items()])
            + "."
        )


def describe_site_placement(surface: Slab, site: tuple, cutoff=2.5):
    """Describe the site placement for the given surface, site, cutoff."""
    n = get_neighbors_site(surface, site, cutoff=cutoff)
    return describe_neighbors_site(n)


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