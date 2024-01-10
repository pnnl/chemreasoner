"""Module for reward funciton by calculation of adsorption energies in simulation."""
import logging
import sys
import uuid
from pathlib import Path
import numpy as np

from ase import Atoms
from ase.io import read
from ase.data import chemical_symbols


sys.path.append("src")
from llm import ase_interface  # noqa: E402
from search.state.reasoner_state import ReasonerState  # noqa: E402
from evaluation.break_traj_files import break_trajectory  # noqa: E402
from nnp import oc  # noqa: E402
from search.reward.base_reward import BaseReward  # noqa: E402

logging.getLogger().setLevel(logging.INFO)


class PathReward(BaseReward):
    """Calculate the reward for answers based on adsorption simulations."""

    def __init__(
        self,
        llm_function: callable,
        penalty_value: float = -10,
        nnp_class="oc",
        num_slab_samples=16,
        num_adslab_samples=16,
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

        generation_results = self.llm_function(
            prompts, system_prompts, **{"temperature": 0.7, "top_p": 0.95}
        )
        loop_counter = 0
        for i, s in enumerate(states):
            if slab_syms[i] is None:
                s.process_generation(generation_results[loop_counter])

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
        print(answers)

        for i, p in enumerate(prompts):
            state_idx = prompts_idx[i]
            s = states[state_idx]
            try:
                print(s.process_catalyst_symbols(answers[i]))
                slab_syms[state_idx] = s.process_catalyst_symbols(answers[i])

            except Exception as err:
                logging.warning(f"Failed to parse answer with error: {str(err)}.")

    def __call__(self, states: list[ReasonerState]):
        """Return the calculated adsorption energy from the predicted catalysts."""

        _, min_act_energy, min_act_energy_path = self.get_reward_for_paths(paths)

        print("minimum activation energy aproximation: ", min_act_energy)
        print("minimum activation energy reaction pathway: ", min_act_energy_path)

        return min_act_energy_path

    def create_structures_and_calculate(
        self,
        slab_syms,
        ads_list,
        candidates_list=None,
        adsorbate_height=1,
        placement_type=None,
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
                    print("slab is not present. creating new one.")
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
                    print("salb is present")
                    if placement_type == None:
                        for ads_sym in ads_list:
                            ads_ats = ase_interface.ads_symbols_to_structure(ads_sym)
                            name = f"{slab_name}_{ads_sym}"
                            adslab_ats += self.sample_adslabs(
                                slab_ats, ads_ats, name, adsorbate_height
                            )
                            if candidates_list is not None:
                                name_candidate_mapping[name] = candidates_list[i]

                    elif placement_type == "heuristic":
                        for ads_sym in ads_list:
                            ads_ats = ase_interface.ads_symbols_to_structure(ads_sym)
                            # slab_ats.center(vacuum=13.0, axis=2)

                            name = f"{slab_name}_{ads_sym}"
                            adslab_ats += self.sample_adslabs_heuristic(
                                slab_ats, ads_ats, name
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
        self,
        adslabs_and_energies,
        name_candidate_mapping,
        candidates_list,
        ads_preferences,
    ):
        """Parse adsorption energies to get the reward value."""
        # Parse out the rewards into candidate/adsorbate
        reward_values = {}
        for idx, name, energy, valid_structure in adslabs_and_energies:
            cand = name_candidate_mapping[name]
            ads = name.split("_")[-1]
            if valid_structure == 0:
                if cand in reward_values.keys():
                    reward_values[cand][ads] += [energy]
                else:
                    reward_values[cand] = {ads: [energy]}
            else:
                if cand not in reward_values.keys():
                    reward_values[cand] = {ads: []}

        # aggregate the rewards
        rewards = []
        for cand in candidates_list:
            if cand in reward_values.keys():
                rewards.append(
                    [
                        -((min(reward_values[cand][ads])) ** ads_preferences[i])
                        if reward_values[cand][ads] > 0
                        else self.penalty_value
                        for i, ads in enumerate(reward_values[cand].keys())
                    ]
                )
            else:  # Handle default here TODO: determine some logic/pentaly for this
                print(cand)
                return rewards.append(self.penalty_value)

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
                    valid = self.adsorption_calculator.get_validity(name, idx)
                    results.append((idx, name, ads_calc, valid))
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

    def get_reward_for_path(self, path: list):
        adsE = []
        # reward_values = defaultdict()
        for j, step in enumerate(path):
            adsorbent, adsorbate, name = step
            # print('step: ', j)

            # use the same adsorbent
            # create the adsorbent here or if the slab is
            # saved, read it.
            res = sr.create_structures_and_calculate(
                [adsorbent], [adsorbate], [name], placement_type="heuristic"
            )

            for p in Path("data", "output", f"{traj_dir}").rglob("*.traj"):
                break_trajectory(p)

            adslabs_and_energies = res[0]  # id, slab_name, ads_energy, valid
            # name_candidate_mapping = res[3]
            # adslabs_and_energies

            # only selecting the valid structures
            adslabs_and_energies = [i for i in adslabs_and_energies if i[3] == 1]

            # get the energies of each adsorbed structures
            energies = [i[2] for i in adslabs_and_energies]

            # get the minimum energy structure
            lowest_E_str = adslabs_and_energies[np.argmin(energies)]

            # print("low E ", lowest_E_str)
            adsE.append(lowest_E_str)

        E = [i[2] for i in adsE]
        print("energy difference between steps: ", np.diff(E))
        max_E_diff = max(np.diff(E))  # an approximation for activation energy

        return max_E_diff
        # return max_E_diff, E

    def get_reward_for_paths(self, paths):
        rewards = []
        # ads_energies=[]
        for path in paths:
            reward = self.get_reward_for_path(path)
            # reward, E = self.get_reward_for_path(path)
            # ads_energies.append(E)
            rewards.append(reward)

        min_act_energy_path_id = np.argmin(rewards)
        min_act_energy = rewards[min_act_energy_path_id]
        min_act_energy_path = paths[min_act_energy_path_id]

        return min_act_energy_path_id, min_act_energy, min_act_energy_path
        # return ads_energies, min_act_energy, min_act_energy_path

    def unpack_batch_results(self, batch_results, fname_batch):
        """Unpack a collection of batch results."""
        results = []
        for i, res in enumerate(batch_results):
            idx = Path(fname_batch[i]).stem.split("-")[0]
            name = str(Path(fname_batch[i]).parent)
            valid = self.adsorption_calculator.get_validity(name, idx)
            results.append((idx, name, res, valid))
        return results

    def calculate_batch(self, adslab_batch, fname_batch):
        """Calculate adsorption energies for a batch of atoms objects."""
        batch_relaxed = self.adsorption_calculator.batched_relax_atoms(
            atoms=adslab_batch, atoms_names=fname_batch
        )
        batch_adsorption_energies = (
            self.adsorption_calculator.batched_adsorption_calculation(
                atoms=batch_relaxed, atoms_names=fname_batch
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

    def sample_adslabs_heuristic(self, slab, ads, name):
        """Sample possible adsorbate+slab combinations."""
        adslabs = []
        # for i in range(self.num_adslab_samples):
        # print(slab.info)
        adslab = ase_interface.generate_bulk_ads_pairs_heuristic(
            slab, ads, num_sites=self.num_adslab_samples
        )
        adslabs = [(i, name, adslab[i]) for i in range(len(adslab))]

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


# class _TestState:
#     def __init__(self, test_candidates, test_ads_symbols, test_ads_preferences):
#         """Create test query state for testing the reward function."""
#         self.candidates = test_candidates
#         self.ads_symbols = test_ads_symbols
#         self.ads_preferences = test_ads_preferences


if __name__ == "__main__":
    traj_dir = "hreact"

    sr = PathReward(
        **{
            "llm_function": None,
            "model": "gemnet",
            "traj_dir": Path("data", "output", f"{traj_dir}"),
            "device": "cpu",
            "ads_tag": 2,
            "num_adslab_samples": 2,
        }
    )

    path1 = [
        [["Cu"], "CO2", "Cu"],
        [["Cu"], "HCOOH", "Cu"],
        [["Cu"], "CH2O", "Cu"],
        [["Cu"], "CH3O", "Cu"],
        [["Cu"], "CH3OH", "Cu"],
    ]

    path2 = [
        [["Cu"], "CO2", "Cu"],
        [["Cu"], "CO", "Cu"],
        [["Cu"], "CHOH", "Cu"],
        [["Cu"], "CH3OH", "Cu"],
    ]

    path3 = [
        [["Cu"], "CO2", "Cu"],
        [["Cu"], "CHO2", "Cu"],
        [["Cu"], "CH3OH", "Cu"],
        [["Cu"], "CH2O", "Cu"],
        [["Cu"], "CH3OH", "Cu"],
    ]

    path4 = [
        [["Cu"], "CO2", "Cu"],
        [["Cu"], "COOH", "Cu"],
        [["Cu"], "CH2O", "Cu"],
        [["Cu"], "CH3OH", "Cu"],
    ]

    path5 = [[["Cu"], "CO2", "Cu"], [["Cu"], "CHO", "Cu"], [["Cu"], "CH3OH", "Cu"]]

    path_id, actE, path = sr.get_reward_for_paths([path1, path2, path4, path5])

    print("minimum act. energy: ", actE)
    print("minimum act. energy path : ", path)
