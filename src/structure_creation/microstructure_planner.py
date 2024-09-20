"""Moduel for microstructure creation from LLM answers for the OCP format."""

import logging
import os
import pickle
import sys
import time

from ast import literal_eval
from copy import deepcopy
from pathlib import Path
from typing import Optional

from ase.data import chemical_symbols
from ase.io import write

import crystal_toolkit

import mp_api
from pymatgen.core.surface import SlabGenerator
from pymatgen.ext.matproj import MPRester

import numpy as np
import pandas as pd
from ocdata.core import Slab, Adsorbate, AdsorbateSlabConfig

sys.path.append("src")
from llm.utils import process_prompt
from llm.azure_open_ai_interface import AzureOpenaiInterface
from search.state.reasoner_state import ReasonerState
from structure_creation.digital_twin import CatalystDigitalTwin

logging.getLogger().setLevel(logging.INFO)

MP_API_KEY = os.environ["MP_API_KEY"]

with open(Path("data", "input_data", "oc", "oc_20_adsorbates.pkl"), "rb") as f:
    oc_20_ads_structures = pickle.load(f)
    oc_20_ads_structures = {
        v[1]: (v[0], v[2:]) for k, v in oc_20_ads_structures.items()
    }

print(oc_20_ads_structures["*CO"])

prompts = {
    "bulk": {
        "prompt": (
            r"$ROOT_PROMPT = '{root_prompt}'\n\nConsider the following list of materials. \nReturn the index of the material that would be best suited for answering the $ROOT_PROMPT.\n\n{bulks_summaries}\nReturn your answer as a python list called 'final_answer' of your top {num_choices} choices, using the indices given in the () above. Let's think step-by-step and provide justifications for your answers. Remember to return a python list called final answer!"
        ),
        "system_prompt": (
            "You are an AI assistant that has knowledge about materials science and can make accurate recommendations about bulk material structures based on their crystal structure and composition. You will consider factors such as catalytic performance and synthesizability in your analysis."
        ),
    },
    "millers": {
        "prompt": (
            r"$ROOT_PROMPT = '{root_prompt}'\n\nConsider the material {material}. \nReturn a list of miller indices that would answer the $ROOT_PROMPT. You may consider high as well as low index planes. \n\n\nReturn your answer as a python list called 'final_answer' of your top {num_choices} miller indices. Your miller indices should be python 3-tuples. Let's think step-by-step and provide justifications for your answers. Remember to return a python list called final answer!"
        ),
        "system_prompt": (
            "You are an AI assistant that has knowledge about materials science and catalysis and can make accurate recommendations about surface composition based on surface chemistry. You will consider factors such as catalytic performance and binding sites."
        ),
    },
    "surface": {
        r"prompt": "$ROOT_PROMPT = {root_prompt}\n\nConsider the material {material} with miller index {millers}. Return the index of the following surfaces which has the best configuration for accomplishing the $ROOT_PROMPT. \n\n{cell_shift_summaries}\n\nReturn your answer as a python list called 'final_answer' of your top {num_choices} choices, using the indices given in the () above. Let's think step-by-step and provide justifications for your answers. Remember to return a python list called final answer!",
        "system_prompt": (
            "You are an AI assistant that has knowledge about materials science and catalysis and can make accurate recommendations about miller indices based on surface chemistry. You will consider factors such as catalytic performance and binding sites."
        ),
    },
    "site_placement": {
        "prompt": (
            r"$ROOT_PROMPT = '{root_prompt}'\n\nConsider the material {material} with miller index {millers}. Return the indices of the atomic environment of the best adsorbate placement sites.\n\n{atomic_environments}\nReturn your answer as a python list called 'final_answer' of your top {num_choices} choices, using the indices given in the () above. Let's think step-by-step and provide justifications for your answers. Remember to return a python list called final answer!"
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
        "bulk": 2,
        "millers": 2,
        "site_placement": 10,
        "__default__": 1,
    }
    num_choices = {
        "bulk": 4,
        "millers": 4,
        "site_placement": 4,
    }

    def __init__(
        self, llm_function=callable, debug: bool = False, num_choices: dict = {}
    ):
        """Init self."""
        self.llm_function = llm_function
        self._site_placements_indices = {}
        self.num_choices.update({k: v for k, v in num_choices.items() if v is not None})

    def update_num_choices(self, num_choices: dict[str, int]):
        """Update the num_choices in self with given dictionary."""
        self.num_choices.update(num_choices)

    def set_state(self, state: ReasonerState):
        """Set the state for self."""
        self.state = state

    def get_twin_states(self, digital_twins: list[CatalystDigitalTwin]):
        """Get twin-state pair list for given list of twins."""
        return [(d, self.state) for d in digital_twins]

    def set_digital_twins(self, twins: list[CatalystDigitalTwin]):
        """Set digital twins for self."""
        self.digital_twins = deepcopy(twins)

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

    @staticmethod
    def literal_parse_response_list(response: str) -> list[int]:
        """Create a prompt for the given dictionaries."""
        final_answer_idx = response.find("final_answer")
        list_start = response.find("[", final_answer_idx)
        list_end = response.find("]", list_start)

        answer_list = literal_eval(response[list_start : list_end + 1])
        return answer_list

    def create_bulk_prompt(self, twin_state: tuple[CatalystDigitalTwin, ReasonerState]):
        """Create the prompt for bulks."""
        twin, state = twin_state
        bulks = twin.get_bulks()
        bulks_summaries = ""
        for i, doc in enumerate(bulks):
            verified = " (experimentally verified)" if not doc.theoretical else ""
            bulks_summaries += f"({i}) {doc.formula_pretty} in the {doc.symmetry.crystal_system.value.lower()} {doc.symmetry.symbol} space group{verified}.\n"

        prompt_values = {
            "bulks_summaries": bulks_summaries,
            "root_prompt": state.root_prompt,
            "num_choices": self.num_choices["bulk"],
        }
        prompt = fstr(prompts["bulk"]["prompt"], prompt_values)
        twin.update_info("bulk", {"prompt": prompt}, start_new=True)
        return prompt

    def create_bulk_system_prompt(
        self, twin_state: tuple[CatalystDigitalTwin, ReasonerState]
    ):
        """Create millers system prompt."""
        twin, state = twin_state
        prompt = prompts["bulk"]["system_prompt"]
        twin.update_info("bulk", {"system_prompt": prompt})
        return prompt

    def parse_bulk_answer(
        self, answer_data, twin_state: tuple[CatalystDigitalTwin, ReasonerState]
    ):
        """Parse the bulk_prompt_response."""
        # TODO: Track the behavior here
        twin, state = twin_state
        twin.update_info("bulk", answer_data)
        answer_list = self.literal_parse_response_list(answer_data["answer"])

        return answer_list

    def run_bulk_prompt(self, digital_twins: list[CatalystDigitalTwin]):
        """Run the bulk prompt for the given slab symbols."""
        twin_states = self.get_twin_states(digital_twins)
        bulks_idxs = self.process_prompt(
            prompt_info=twin_states,
            prompt_type="bulk",
            prompt_creation_function=self.create_bulk_prompt,
            prompt_parsing_function=self.parse_bulk_answer,
            system_prompt_function=self.create_bulk_system_prompt,
            # TODO: LLM function kwargs
        )
        return bulks_idxs

    def create_millers_prompt(
        self, twin_state: tuple[CatalystDigitalTwin, ReasonerState]
    ):
        """Create a prompt for the miller index."""
        twin, state = twin_state
        doc = twin.computational_objects["bulk"]
        values = {
            "root_prompt": state.root_prompt,
            # "answer": state.answer,
            "material": f"{doc.formula_pretty} in the {doc.symmetry.crystal_system.value.lower()} {doc.symmetry.symbol} space group.\n",
            "num_choices": self.num_choices["millers"],
        }
        prompt = fstr(prompts["millers"]["prompt"], values)
        twin.update_info("millers", {"prompt": prompt}, start_new=True)
        return prompt

    def create_millers_system_prompt(
        self, twin_state: tuple[CatalystDigitalTwin, ReasonerState]
    ):
        """Create millers system prompt."""
        twin, state = twin_state
        prompt = prompts["millers"]["system_prompt"]
        twin.update_info("millers", {"system_prompt": prompt})
        return prompt

    def parse_millers_answer(
        self, answer_data, twin_state: tuple[CatalystDigitalTwin, ReasonerState]
    ):
        """Parse the given answer for the miller indices."""
        twin, state = twin_state
        twin.update_info("millers", answer_data)
        print(answer_data["answer"])
        try:
            answer_list = self.literal_parse_response_list(answer_data["answer"])
        except Exception:
            # Try manually parsing out numbers
            response = answer_data["answer"]
            final_answer_idx = response.find("final_answer")
            list_start = response.find("[", final_answer_idx)
            list_end = response.find("]", list_start)

            answer_list = []
            these_miller_indices = []  # Assume miller index is always single digits
            minus_1 = 1
            for char in response[list_start : list_end + 1]:
                if char == ",":
                    answer_list.append(tuple(these_miller_indices))
                    these_miller_indices = []
                print(char)
                if char == "-":
                    minus_1 = -1
                elif char.isnumeric():  # assume miller index always single digit
                    these_miller_indices.append(minus_1 * int(char))
                    minus_1 = 1

        print(answer_list)
        return answer_list

    def run_millers_prompt(self, digital_twins: CatalystDigitalTwin):
        """Run the bulk prompt for the given slab symbols."""
        twin_states = self.get_twin_states(digital_twins)
        millers_choices = self.process_prompt(
            twin_states,
            "millers",
            self.create_millers_prompt,
            self.parse_millers_answer,
            self.create_millers_system_prompt,
            # TODO: LLM function kwargs
        )
        return millers_choices

    def create_site_placement_prompt(
        self, twin_state: tuple[CatalystDigitalTwin, ReasonerState]
    ):
        """Create the prompt for site_placement."""
        twin, state = twin_state
        site_placements = twin.get_site_placements()
        _identified_sites = []
        self._site_placements_indices[twin._id] = {}
        if len(site_placements) > 0:
            site_placements_summaries = []
            j = 0
            for i, site in enumerate(site_placements):
                description = describe_site_placement(
                    twin.computational_objects["surface"], site
                )
                if description not in _identified_sites:
                    _identified_sites.append(description)
                    self._site_placements_indices[twin._id][j] = [i]
                    j += 1
                else:
                    j_idx = _identified_sites.index(description)
                    self._site_placements_indices[twin._id][j_idx].append(i)

            for j, desc in enumerate(_identified_sites):
                site_placements_summaries.append(
                    f"({j}) {desc}"
                )  # TODO: include {len(self._site_placements_indices[j])}
        else:
            return None
        doc = twin.computational_objects["bulk"]
        values = {
            "root_prompt": state.root_prompt,
            "material": f"{doc.formula_pretty} in the {doc.symmetry.crystal_system.value.lower()} {doc.symmetry.symbol} space group.\n",
            "millers": f"{twin.computational_params['millers']}",
            "atomic_environments": "\n".join(site_placements_summaries),
            "num_choices": self.num_choices["site_placement"],
        }
        prompt = fstr(prompts["site_placement"]["prompt"], values)
        twin.update_info("site_placement", {"prompt": prompt}, start_new=True)
        return prompt

    @staticmethod
    def create_site_placement_system_prompt(
        twin_state: tuple[CatalystDigitalTwin, ReasonerState]
    ):
        """Create the prompt for site_placement."""
        twin, state = twin_state
        prompt = prompts["site_placement"]["system_prompt"]
        twin.update_info("site_placement", {"system_prompt": prompt})
        return prompt

    def parse_site_placement_answer(
        self, answer_data, twin_state: tuple[CatalystDigitalTwin, ReasonerState]
    ):
        """Parse the given answer for the miller indices."""
        twin, state = twin_state
        twin.update_info("site_placement", answer_data)
        answer_list = self.literal_parse_response_list(answer_data["answer"])

        return answer_list

    def run_site_placement_prompt(self, digital_twins: CatalystDigitalTwin):
        """Run the bulk prompt for the given slab symbols."""
        print()
        twin_states = self.get_twin_states(digital_twins)
        site_choices = self.process_prompt(
            twin_states,
            "site_placement",
            self.create_site_placement_prompt,
            self.parse_site_placement_answer,
            self.create_site_placement_system_prompt,
            # TODO: LLM function kwargs
        )
        site_choices = self.get_site_indices(digital_twins, site_choices)
        return site_choices

    def get_site_indices(
        self, digital_twins: CatalystDigitalTwin, site_choices: CatalystDigitalTwin
    ):
        """Get the actual indices of chosen sites."""
        new_site_choices = []
        for twin, site_choice in zip(digital_twins, site_choices):
            chosen_indices = []
            for s in site_choice:
                chosen_indices += self._site_placements_indices[twin._id][s]
            these_sites = twin.get_site_placements()
            new_site_choices.append([these_sites[idx] for idx in chosen_indices])

        return new_site_choices


def get_neighbors_site(surface: Slab, site: tuple, cutoff=2.5):
    """Get the neighboring atoms of the given site."""
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
    prompt_templates = prompts["bulk"]
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
        "bulk": "mp-2133",
        "millers": (1, 2, 3),
        "cell_shift": 3,
        "site_placement": (3.2, 4.5, 6.7),
        "atoms_object_id": "{db_id}_{structure_id}",
    }
]


class TestState:
    root_prompt = "Propose a catalyst for the adsorption of *CO."


if __name__ == "__main__":
    start = time.time()
    llm_function = AzureOpenaiInterface(dotenv_path=".env", model="gpt-4")
    ms_planner = OCPMicrostructurePlanner(llm_function=llm_function)

    state = TestState()
    ms_planner.set_state(state)

    dt = CatalystDigitalTwin()
    dt.set_symbols(["Cu", "Zn"])
    digital_twins = [dt]
    ms_planner.run_bulk_prompt(digital_twins)

    ms_planner.run_millers_prompt(digital_twins=digital_twins)
    for t in digital_twins:
        print(t.info)

    num_twins = len(digital_twins)
    for i in range(num_twins):
        print(digital_twins[i].get_surfaces())
        digital_twins += digital_twins[i].set_surfaces(
            digital_twins[i].get_surfaces()[0]
        )

    ms_planner.run_site_placement_prompt(digital_twins=digital_twins)

    save_dir = Path("microstructure_planner_test")
    save_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    row_data = []

    for twin in digital_twins:
        _id = twin._id
        row = twin.return_row()

        ads_ats, binding_atoms = oc_20_ads_structures["*CO"]
        adsorbate = Adsorbate(ads_ats, adsorbate_binding_indices=list(binding_atoms))

        adslab_config = twin.return_adslab_config(adsorbate=adsorbate)
        adslab = adslab_config.atoms_list[0]
        adslab.info.update({"id": str(_id)})
        adslab_metadata = adslab_config.metadata_list[0]

        row.update(adslab_metadata)
        row_data.append(row)

        write(str(save_dir / f"{_id}.xyz"), adslab)

        metadata[_id] = twin.info
    end = time.time()
    print(f"TIME: {end-start}")
    with open(save_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    df = pd.DataFrame(row_data)
    df.to_csv(save_dir / "row_data.csv")
