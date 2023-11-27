"""Module for reward funciton by calculation of adsorption energies in simulation."""
import logging
import sys
import uuid

from pathlib import Path

import numpy as np

from ase import Atoms
from ase.io import read
from ase.neighbor_list import build_neighbor_list
from ase.data import chemical_symbols


sys.path.append("src")
from llm import ase_interface  # noqa: E402
from search.state.reasoner_state import ReasonerState  # noqa: E402
from evaluation.break_traj_files import break_trajectory  # noqa: E402
from nnp import oc  # noqa: E402
from search.reward.base_reward import BaseReward  # noqa: E402

logging.getLogger().setLevel(logging.INFO)


class StructureReward(BaseReward):
    """Calculate the reward for answers based on adsorption simulations."""

    def __init__(
        self,
        llm_function: callable,
        penalty_value: float = -10,
        nnp_class="oc",
        num_slab_samples=8,
        num_adslab_samples=8,
        **nnp_kwargs,
    ):
        """Select the class of nnp for the reward function."""
        self.llm_function = llm_function
        self.penalty_value = penalty_value
        if nnp_class == "oc":
            self.adsorption_calculator = oc.OCAdsorptionCalculator(**nnp_kwargs)
        else:
            raise NotImplementedError(f"No such nnp class {nnp_class}.")
        self.num_slab_samples = num_slab_samples
        self.num_adslab_samples = num_adslab_samples

    def run_generation_prompts(
        self, slab_syms: list[list[str]], states: list[ReasonerState]
    ):
        """Run the generation prompts for the given states where the reward is None."""
        prompts = []
        system_prompts = []
        for i, s in enumerate(states):
            if slab_syms[i] is None:
                prompts.append(s.generation_prompt)
                system_prompts.append(s.generation_system_prompt)

        generation_answers = self.llm_function(prompts, system_prompts)
        loop_counter = 0
        for i, s in enumerate(states):
            if slab_syms[i] is None:
                s.process_generation(generation_answers[loop_counter])

                loop_counter += 1

    def run_slab_sym_prompts(
        self, slab_syms: list[list[str]], states: list[ReasonerState]
    ):
        """Run the generation prompts for the given states where the reward is None.

        Updates the given "slab_syms" list in-place.
        """
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

        answers = self.llm_function(
            prompts, system_prompts, **{"temperature": 0.0, "top_p": 0}
        )

        for i, p in enumerate(prompts):
            state_idx = prompts_idx[i]
            s = states[state_idx]
            try:
                slab_syms[state_idx] = s.process_catalyst_symbols(answers[i])

            except Exception as err:
                logging.warning(f"Failed to parse answer with error: {str(err)}.")

    def __call__(
        self,
        states: list[ReasonerState],
        num_attempts: int = 3,
        primary_reward: bool = True,
    ):
        """Return the calculated adsorption energy from the predicted catalysts."""
        rewards = []
        slab_syms = [None] * len(states)
        attempts = 0
        while any([r is None for r in rewards]) and attempts < self.max_attempts:
            if primary_reward:
                self.run_generation_prompts(slab_syms, states)

            self.run_slab_sym_prompts(slab_syms, states)

            attempts += 1

        for i, s in enumerate(states):
            ads_list = s.ads_symbols
            candidates_list = s.candidates
            if slab_syms[i] is None:
                logging.warning(
                    f"Unable to parse the answer:\n\n {s.answer}."
                    "\n\nInto catalyst symbols. "
                    "Returning the penalty value for that answer."
                )
                rewards[i] = self.penalty_value
            else:
                (
                    adslabs_and_energies,
                    gnn_calls,
                    gnn_time,
                    name_candidate_mapping,
                ) = self.create_structures_and_calculate(
                    slab_syms, ads_list, candidates_list
                )

                final_reward = self.parse_adsorption_energies(
                    adslabs_and_energies, name_candidate_mapping, candidates_list
                )

                rewards.append(final_reward)

                s.info["simulation-reward"].update(
                    {
                        "slab_syms": slab_syms,
                        "value": final_reward,
                        "gnn_calls": gnn_calls,
                        "gnn_time": gnn_time,
                    }
                )

        return rewards

    def create_structures_and_calculate(
        self, slab_syms, ads_list, candidates_list=None, adsorbate_height=1
    ):
        """Create the structures from the symbols and calculate adsorption energies."""
        start_gnn_calls = self.adsorption_calculator.gnn_calls
        start_gnn_time = self.adsorption_calculator.gnn_time
        adslab_ats = []  # List to store initial adslabs and indices
        name_candidate_mapping = (
            {}
        )  # dictionary to get from database names to candidates
        for i, slab_sym in enumerate(slab_syms):
            if slab_sym is not None:
                valid_slab_sym = True
                slab_name = self.reduce_candidate_symbols(slab_sym)
                slab_ats = self.adsorption_calculator.get_slab(slab_name)
                if slab_ats is None:
                    try:
                        slab_samples = [
                            ase_interface.symbols_list_to_bulk(slab_sym)
                            for _ in range(self.num_slab_samples)
                        ]
                        print(slab_samples)
                    except ase_interface.StructureGenerationError as err:
                        print(err)
                        slab_syms[i] = None
                        valid_slab_sym = False

                    if valid_slab_sym:
                        slab_ats = self.adsorption_calculator.choose_slab(
                            slab_samples, slab_name
                        )
                if slab_ats is not None:
                    for ads_sym in ads_list:
                        ads_ats = ase_interface.ads_symbols_to_structure(ads_sym)
                        name = f"{slab_name}_{ads_sym}"
                        adslab_ats += self.sample_adslabs(
                            slab_ats, ads_ats, name, adsorbate_height
                        )
                        if candidates_list is not None:
                            name_candidate_mapping[name] = candidates_list[i]

        adslabs_and_energies = self.create_batches_and_calculate(adslab_ats)

        end_gnn_calls = self.adsorption_calculator.gnn_calls
        end_gnn_time = self.adsorption_calculator.gnn_time

        return (
            adslabs_and_energies,
            end_gnn_calls - start_gnn_calls,
            end_gnn_time - start_gnn_time,
            name_candidate_mapping,
        )

    def parse_adsorption_energies(
        self, adslabs_and_energies, name_candidate_mapping, candidates_list
    ):
        """Parse adsorption energies to get the reward value."""
        # Parse out the rewards into candidate/adsorbate
        reward_values = {}
        for idx, name, energy in adslabs_and_energies:
            cand = name_candidate_mapping[name]
            ads = name.split("_")[-1]
            if cand in reward_values.keys():
                if name.split("_")[-1] in reward_values[cand].keys():
                    reward_values[cand][ads] += [energy]
                else:
                    reward_values[cand][ads] = [energy]
            else:
                reward_values[cand] = {ads: [energy]}

        # aggregate the rewards
        rewards = []
        for cand in candidates_list:
            if cand in reward_values.keys():
                rewards.append(
                    np.mean(
                        [
                            -((min(reward_values[cand][ads])) ** s.ads_preferences[i])
                            for i, ads in enumerate(reward_values[cand].keys())
                        ]
                    )
                )
            else:  # Handle default here TODO: determine some logic/pentaly for this
                print(cand)
                return -10

        final_reward = np.mean(rewards)

        return final_reward  # return mean over candidates

    def create_batches_and_calculate(self, adslabs):
        """Split adslabs into batches and run the simulations."""
        results = []
        adslab_batch = []
        fname_batch = []
        for idx, name, adslab in adslabs:
            fname = Path(f"{name}") / f"{idx}"
            (self.adsorption_calculator.traj_dir / fname).parent.mkdir(
                parents=True, exist_ok=True
            )
            if (
                len(
                    list(
                        self.adsorption_calculator.traj_dir.rglob(str(fname) + "*.traj")
                    )
                )
                == 0
            ):
                print("****")
                print(adslab)
                adslab_batch.append(adslab)
                fname_batch.append(str(fname) + f"-{uuid.uuid4()}")
            else:
                idx = str(fname.stem)
                name = str(fname.parent)

                # Get pre calculated values if they exists. Otherwise, create batch
                ads_calc = self.adsorption_calculator.get_prediction(name, idx)
                if ads_calc is not None:
                    results.append((idx, name, ads_calc))
                else:
                    adslab_batch.append(adslab)
                    fname_batch.append(str(fname) + f"-{uuid.uuid4()}")

            # dispatch the batch
            if len(adslab_batch) == self.adsorption_calculator.batch_size:
                batch_results = self.calculate_batch(adslab_batch, fname_batch)
                results += self.unpack_batch_results(batch_results, fname_batch)
                adslab_batch = []
                fname_batch = []
        # dispatch the remaining batch
        if len(adslab_batch) > 0:
            batch_results = self.calculate_batch(adslab_batch, fname_batch)
            results += self.unpack_batch_results(batch_results, fname_batch)
            adslab_batch = []
            fname_batch = []

        return results

    @staticmethod
    def unpack_batch_results(batch_results, fname_batch):
        """Unpack a collection of batch results."""
        results = []
        for i, res in enumerate(batch_results):
            idx = Path(fname_batch[i]).stem.split("-")[0]
            name = str(Path(fname_batch[i]).parent)
            results.append((idx, name, res))
        return results

    def calculate_batch(self, adslab_batch, fname_batch):
        """Calculate adsorption energies for a batch of atoms objects."""
        batch_relaxed = self.adsorption_calculator.batched_relax_atoms(
            adslab_batch, fname_batch
        )
        batch_adsorption_energies = (
            self.adsorption_calculator.batched_adsorption_calculation(
                batch_relaxed, fname_batch
            )
        )
        return batch_adsorption_energies

    def sample_adslabs(self, slab, ads, name, adsorbate_height):
        """Sample possible adsorbate+slab combinations."""
        adslabs = []
        for i in range(self.num_adslab_samples):
            print(slab.info)
            adslab = ase_interface.generate_bulk_ads_pairs(
                slab, ads, height=adsorbate_height
            )
            adslabs.append((i, name, adslab))
        return adslabs

    @staticmethod
    def reduce_metal_symbols(metal_ats: Atoms):
        """Reduce the symbols of metal symbols to a basic form.

        If there are two metals, the more prominant metal is listed first. If there are
        three, the metals are listed in alphabetical order.
        """
        numbers = metal_ats.get_atomic_numbers()
        syms_count = {}
        for num in numbers:
            sym = chemical_symbols[num]
            if sym in syms_count.keys():
                syms_count[sym] += 1
            else:
                syms_count[sym] = 1

        if len(syms_count) == 2:
            k1, k2 = syms_count.keys()
            if syms_count[k1] > syms_count[k2]:
                name_syms = [k1, k2]
            else:
                name_syms = [k2, k1]
        else:
            name_syms = sorted(list(syms_count.keys()))

        formula = "".join(name_syms)
        return formula

    @staticmethod
    def reduce_candidate_symbols(candidate_syms: list[str]):
        """Reduce the symbols of metal symbols to a basic form.

        If there are two metals, the more prominant metal is listed first. If there are
        three, the metals are listed in alphabetical order.
        """
        print(candidate_syms)
        if len(candidate_syms) == 1:
            formula = candidate_syms[0]
        if len(candidate_syms) == 2:
            formula = "".join(candidate_syms)
        else:
            formula = candidate_syms[0] + "".join(sorted(list(candidate_syms)[1:]))

        return formula


class _TestState:
    def __init__(self, test_candidates, test_ads_symbols, test_ads_preferences):
        """Create test query state for testing the reward function."""
        self.candidates = test_candidates
        self.ads_symbols = test_ads_symbols
        self.ads_preferences = test_ads_preferences


class AdsorbedStructureChecker:
    """A class to check whether an adsorbed structure is correct or not.

    Uses convention created by Open Catalysis:
    https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md

    "0 - no anomaly
    1 - adsorbate dissociation
    2 - adsorbate desorption
    3 - surface reconstruction [not implemented in this code]
    4 - incorrect CHCOH placement, appears to be CHCO with a lone, uninteracting, H far
    off in the unit cell [not implemented in this code]"
    """

    all_clear_code = 0
    adsorbate_dissociation_code = 1
    desorption_code = 2
    surface_reconstruction = 3  # unused
    incorrect_CHCOH = 4  # unused

    def __call__(self, ats: Atoms):
        """Check the given structure for errors."""
        if not self.check_dissociation(ats):
            return self.adsorbate_dissociation_code
        elif not self.check_adsorption(ats):
            return self.desorption_code
        else:
            return self.all_clear_code

    def check_adsorption(self, ats: Atoms):
        """Mesure whether or not the atoms adsorbed"""
        return self.check_connectivity(ats)

    @staticmethod
    def measure_adsorption_distance(ats: Atoms, cutoff=2.0) -> float:
        """Determine whether the adsorbate has adsorbed."""
        D = ats.get_all_distances()
        adsorbate_ats = ats.get_tags() == 0
        # return not any(
        #     np.any(
        #         np.less(D[np.ix_(adsorbate_ats, ~adsorbate_ats)], cutoff),
        #         axis=1,
        #     )
        # )
        return min(D[np.ix_(adsorbate_ats, ~adsorbate_ats)].flatten())

    def measure_dissociation(self, ats: Atoms):
        """Determine whether the adsorbate has dissociated."""
        idx = ats.get_tags() == 0
        ads_atoms = Atoms(
            symbols=ats.get_atomic_numbers()[idx], positions=ats.get_positions()[idx]
        )
        ads_atoms.set_cell(ats.get_cell())

        return self.check_connectivity(ads_atoms)

    @staticmethod
    def check_connectivity(ats: Atoms):
        """Check the connectivity matrix of the given atoms."""
        conn_matrix = build_neighbor_list(ats).get_connectivity_matrix()
        return all(conn_matrix, all)


if __name__ == "__main__":
    heights = np.arange(0.1, 3.0, 0.25)
    for height in heights:
        sr = StructureReward(
            **{
                "model": "gemnet",
                "traj_dir": Path("data", "output", f"adsorption_testing_{height}"),
                "device": "cuda:0",
            }
        )
        print(
            sr.create_structures_and_calculate(
                [["Cu"], ["Pt"], ["Zr"]],
                ["CO", "phenol", "anisole"],
                ["Cu", "Pt", "Zr"],
                adsorbate_height=height,
            )
        )
        for p in Path("data", "output", "adsorption_testing").rglob("*.traj"):
            break_trajectory(p)
