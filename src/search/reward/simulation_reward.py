"""Module for reward function by calculation of adsorption energies in simulation."""

import json
import logging
import requests
import shutil
import sys
import time
import uuid

from copy import deepcopy
from pathlib import Path
from traceback import format_exc

import numpy as np
import torch

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

print("finished imports")


class StructureReward(BaseReward):
    """Calculate the reward for answers based on adsorption simulations."""

    def __init__(
        self,
        llm_function: callable,
        penalty_value: float = -10,
        nnp_class="oc",
        num_slab_samples=16,
        num_adslab_samples=16,
        max_attempts: int = 3,
        gnn_service_port: int = None,
        flip_negative: bool = False,
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
        self.max_attempts = max_attempts
        self.gnn_service_port = gnn_service_port
        self.minus = -1 if flip_negative else 1

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
    ):
        """Return the calculated adsorption energy from the predicted catalysts."""
        rewards = []
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

        for i, s in enumerate(states):
            ads_list = s.ads_symbols
            candidates_list = s.candidates
            if slab_syms[i] is None:
                logging.warning(
                    f"Unable to parse the answer:\n\n {s.answer}."
                    "\n\nInto catalyst symbols. "
                    "Returning the penalty value for that answer."
                )
                gnn_calls = 0
                gnn_time = 0
                final_reward = self.penalty_value
            else:
                start = time.time()
                print(self.gnn_service_port)
                if self.gnn_service_port is not None:
                    # Go to the gnn server to get adsorption energy calculations

                    (
                        adslabs_and_energies,
                        gnn_calls,
                        gnn_time,
                        gnn_relaxed,
                        energies_retrieved,
                        name_candidate_mapping,
                    ) = self.call_gnn_server(
                        slab_syms[i],
                        ads_list,
                        candidates_list,
                    )
                else:
                    (
                        adslabs_and_energies,
                        gnn_calls,
                        gnn_time,
                        gnn_relaxed,
                        energies_retrieved,
                        name_candidate_mapping,
                    ) = self.create_structures_and_calculate(
                        slab_syms[i],
                        ads_list,
                        candidates_list,
                    )
                end = time.time()
                logging.info(f"TIMING: GNN calculations done {end - start}")
                if s.ads_preferences is not None:
                    final_reward, reward_values = self.parse_adsorption_energies(
                        s,
                        adslabs_and_energies,
                        name_candidate_mapping,
                        candidates_list,
                        s.ads_preferences,
                    )
                else:
                    (
                        final_reward,
                        reward_values,
                        adsorption_energies,
                    ) = self.parse_adsorption_pathways(
                        adslabs_and_energies,
                        name_candidate_mapping,
                        candidates_list,
                        s.reaction_pathways,
                    )

            rewards.append(final_reward)
            if "simulation-reward" in s.info.keys():
                s.info["simulation-reward"].update(
                    {
                        "slab_syms": slab_syms[i],
                        "value": final_reward,
                        "reward_values": deepcopy(reward_values),
                        "gnn_calls": gnn_calls,
                        "gnn_time": gnn_time,
                        "gnn_relaxed": gnn_relaxed,
                        "energies_retrieved": energies_retrieved,
                    }
                )
            else:
                s.info["simulation-reward"] = {
                    "slab_syms": slab_syms[i],
                    "value": final_reward,
                    "reward_values": deepcopy(reward_values),
                    "gnn_calls": gnn_calls,
                    "gnn_time": gnn_time,
                    "gnn_relaxed": gnn_relaxed,
                    "energies_retrieved": energies_retrieved,
                }
            if s.ads_preferences is None:
                s.info["simulation-reward"].update(
                    {"intermediate_energies": adsorption_energies}
                )

        return rewards

    def call_gnn_server(
        self,
        slab_syms,
        ads_list,
        candidates_list,
    ):
        """create_function to call gnn_server."""
        json_args = {
            "slab_syms": slab_syms,
            "ads_list": ads_list,
            "candidates_list": candidates_list,
        }
        url = f"http://localhost:{self.gnn_service_port}/GemNet"
        response = requests.post(url, json=json_args)
        response_dict = response.json()
        return (
            response_dict["adslabs_and_energies"],
            response_dict["gnn_calls"],
            response_dict["gnn_time"],
            response_dict["name_candidate_mapping"],
        )

    def create_structures_and_calculate(
        self,
        slab_syms,
        ads_list,
        candidates_list=None,
        adsorbate_height=1.87,
        placement_type=None,
    ):
        """Create the structures from the symbols and calculate adsorption energies."""
        start_gnn_calls = self.adsorption_calculator.gnn_calls
        start_gnn_time = self.adsorption_calculator.gnn_time
        start_gnn_relaxed = self.adsorption_calculator.gnn_relaxed
        start_energies_retrieved = self.adsorption_calculator.energies_retrieved
        adslab_ats = []  # List to store initial adslabs and indices
        name_candidate_mapping = (
            {}
        )  # dictionary to get from database names to candidates
        for i, slab_sym in enumerate(slab_syms):
            logging.info(slab_sym)
            try:
                if slab_sym is not None:
                    valid_slab_sym = True
                    slab_name = self.reduce_candidate_symbols(slab_sym)
                    slab_ats = self.adsorption_calculator.get_slab(slab_name)
                    if slab_ats is None:
                        try:
                            if any(
                                [
                                    s not in chemical_symbols
                                    or chemical_symbols.index(s) > 82
                                    for s in slab_sym
                                ]
                            ):
                                raise ase_interface.StructureGenerationError(
                                    f"Cannot create bulk with slab_syms {slab_sym}."
                                )
                            slab_samples = [
                                ase_interface.symbols_list_to_bulk(slab_sym)
                                for _ in range(self.num_slab_samples)
                            ]
                        except ase_interface.StructureGenerationError as err:
                            logging.warning(err)
                            slab_syms[i] = None
                            valid_slab_sym = False

                        if valid_slab_sym:
                            slab_ats = self.adsorption_calculator.choose_slab(
                                slab_samples, slab_name
                            )
                    if slab_ats is not None:
                        if placement_type is None:
                            for ads_sym in ads_list:
                                ads_ats = ase_interface.ads_symbols_to_structure(
                                    ads_sym
                                )
                                name = f"{slab_name}_{ads_sym}"
                                adslab_ats += self.sample_adslabs(
                                    slab_ats, ads_ats, name, adsorbate_height
                                )
                                if candidates_list is not None:
                                    name_candidate_mapping[name] = candidates_list[i]

                        elif placement_type == "heuristic":
                            for ads_sym in ads_list:
                                ads_ats = ase_interface.ads_symbols_to_structure(
                                    ads_sym
                                )
                                # slab_ats.center(vacuum=13.0, axis=2)

                                name = f"{slab_name}_{ads_sym}"
                                adslab_ats += self.sample_adslabs_heuristic(
                                    slab_ats, ads_ats, name
                                )

                                if candidates_list is not None:
                                    name_candidate_mapping[name] = candidates_list[i]
                        else:
                            raise ValueError(f"Unkown placement type {placement_type}.")
            except Exception as err:
                raise err
                logging.warning(
                    f"ERROR:Simulation reward failed for slab syms {slab_syms}. Moving on to the next node."
                )

        adslabs_and_energies = self.create_batches_and_calculate(adslab_ats)

        end_gnn_calls = self.adsorption_calculator.gnn_calls
        end_gnn_time = self.adsorption_calculator.gnn_time
        end_gnn_relaxed = self.adsorption_calculator.gnn_relaxed
        end_energies_retrieved = self.adsorption_calculator.energies_retrieved

        return (
            adslabs_and_energies,
            end_gnn_calls - start_gnn_calls,
            end_gnn_time - start_gnn_time,
            end_gnn_relaxed - start_gnn_relaxed,
            end_energies_retrieved - start_energies_retrieved,
            name_candidate_mapping,
        )

    def parse_adsorption_energies(
        self,
        state,
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
            if valid_structure == 0 or (
                ads[1] == 2 and state.get_ads_preferences(ads) < 0
            ):
                if cand in reward_values.keys():
                    if ads in reward_values.keys():
                        reward_values[cand][ads] += [(energy)]
                    else:
                        reward_values[cand][ads] = [(energy)]
                else:
                    reward_values[cand] = {ads: [(energy)]}
            else:
                if cand not in reward_values.keys():
                    reward_values[cand] = {ads: []}

        # aggregate the rewards
        rewards = []
        for cand in candidates_list:
            if cand in reward_values.keys():
                print(cand, ads)
                print((reward_values[cand][ads]))
                rewards.append(
                    sum(
                        [
                            (
                                -(
                                    (min(reward_values[cand][ads]))
                                    * state.get_ads_preferences(ads)
                                )
                                if len(reward_values[cand][ads]) > 0
                                else self.penalty_value
                            )
                            for i, ads in enumerate(reward_values[cand].keys())
                        ]
                    )
                )
            else:  # Handle default here TODO: determine some logic/pentaly for this
                print(cand)
                rewards.append(self.penalty_value)

        final_reward = np.mean(rewards)

        return final_reward, reward_values  # return mean over candidates

    def parse_adsorption_pathways(
        self,
        adslabs_and_energies,
        name_candidate_mapping,
        candidates_list,
        pathways,
    ):
        """Parse adsorption energies to get the reward value."""
        # Parse out the rewards into candidate/adsorbate
        reward_values = {}
        for idx, name, energy, valid_structure in adslabs_and_energies:
            cand = name_candidate_mapping[name]
            ads = name.split("_")[-1]
            if valid_structure == 0 or (
                ads[1] == 2 and state.get_ads_preferences(ads) < 0
            ):
                if cand in reward_values.keys():
                    if ads in reward_values.keys():
                        reward_values[cand][ads] += [(energy)]
                    else:
                        reward_values[cand][ads] = [(energy)]
                else:
                    reward_values[cand] = {ads: [(energy)]}
            else:
                if cand not in reward_values.keys():
                    reward_values[cand] = {ads: []}

        # aggregate the rewards
        rewards = []
        adsorption_energies = {}
        for cand in candidates_list:
            if cand in reward_values.keys():
                adsorption_energies[cand] = [[None] * len(p) for p in pathways]
                for i, path in enumerate(pathways):
                    for j, ads in enumerate(path):
                        adsorption_energies[cand][i][j] = (
                            min(reward_values[cand][ads])
                            if len(reward_values[cand][ads]) > 0
                            else None
                        )
                paths_without_none = [
                    p for p in adsorption_energies[cand] if None not in p
                ]
                if len(paths_without_none) != 0:
                    reduce_pathways = [
                        max(np.diff(path)) for path in paths_without_none
                    ]
                    rewards.append(self.minus * min(reduce_pathways))
                else:
                    rewards.append(self.penalty_value)

            else:  # Handle default here TODO: determine some logic/pentaly for this
                print(cand)
                rewards.append(self.penalty_value)

        final_reward = np.mean(rewards)

        return (
            final_reward,
            reward_values,
            adsorption_energies,
        )  # return mean over candidates

    def create_batches_and_calculate(self, adslabs):
        """Split adslabs into batches and run the simulations."""
        results = []
        adslab_batch = []
        fname_batch = []
        for idx, name, adslab in adslabs:
            fname = Path(f"{name}") / f"{idx}"
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
                (self.adsorption_calculator.traj_dir / fname).parent.mkdir(
                    parents=True, exist_ok=True
                )

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


if __name__ == "__main__":
    # redis_db = redis.Redis(host='localhost', port=6379, db=0)
    # redis_db.set("/test/thing", "chemreasoner")
    # logging.info(redis_db.get("/test/thing"))
    # # traj_dir = "random"
    # # traj_dir = "heuristic"

    # for model in ["gemnet-t"]:
    #     logging.info("running first...")

    #     start = time.time()
    # sr = StructureReward(
    #     **{
    #         "llm_function": None,
    #         "model": "gemnet-t",
    #         "traj_dir": Path("/dev/shm/chemreasoner/catalysis"),
    #         "device": "cuda",
    #         "steps": 10,
    #         "ads_tag": 2,
    #         "batch_size": 40,
    #         "num_adslab_samples": 2,
    #         "gnn_service_port": None,
    #     }
    # )

    # (
    #     adslabs_and_energies,
    #     gnn_calls,
    #     gnn_time,
    #     name_candidate_mapping,
    # ) = sr.create_structures_and_calculate(
    #     [["Cu", "Al", "Zn"]],
    #     ["CO2", "*CHOH", "*OCHO", "*OHCH3"],
    #     ["CuAlZn"],
    # )

    # name_candidate_mapping = {"CuAlZn": "CuAlZn"}

    # reward_values = {}
    # for idx, name, energy, valid_structure in adslabs_and_energies:
    #     cand = "CuAlZn"
    #     ads = name.split("_")[-1]
    #     if valid_structure == 0 or (ads[1] == 2 and True):
    #         if cand in reward_values.keys():
    #             if ads in reward_values.keys():
    #                 reward_values[cand][ads] += [(energy)]
    #             else:
    #                 reward_values[cand][ads] = [(energy)]
    #         else:
    #             reward_values[cand] = {ads: [(energy)]}
    #     else:
    #         if cand not in reward_values.keys():
    #             reward_values[cand] = {ads: []}

    # # aggregate the rewards
    # rewards = []
    # for cand in ["CuAlZn"]:
    #     if cand in reward_values.keys():
    #         print(cand, ads)
    #         print((reward_values[cand][ads]))
    #         rewards.append(
    #             sum(
    #                 [
    #                     -((min(reward_values[cand][ads])) * 1)
    #                     if len(reward_values[cand][ads]) > 0
    #                     else -10
    #                     for i, ads in enumerate(reward_values[cand].keys())
    #                 ]
    #             )
    #         )
    #     else:  # Handle default here TODO: determine some logic/pentaly for this
    #         print(cand)
    #         rewards.append(-10)

    # final_reward = np.mean(rewards)

    #     end = time.time()
    #     logging.info(end - start)

    #     torch.cuda.empty_cache()

    # for model in ["gemnet-t"]:
    #     logging.info("running second...")

    #     start = time.time()
    # sr = StructureReward(
    #     **{
    #         "llm_function": None,
    #         "model": model,
    #         "traj_dir": Path(f"/dev/shm/testing-gnn/{model}"),
    #         "device": "cuda",
    #         "steps": 64,
    #         "ads_tag": 2,
    #         "batch_size":40,
    #         "num_adslab_samples": 16,
    #     }
    # )

    #     print(
    #         sr.create_structures_and_calculate(
    #             [["Cu"], ["Zn"]],
    #             ["CO2", "*CO"],
    #             ["Cu", "Zn"],
    #             placement_type=None,
    #         )
    #     )

    #     end = time.time()
    #     logging.info(end - start)

    #     torch.cuda.empty_cache()

    for p in Path("src", "nnp", "oc_eval_set").rglob("*/*.traj"):
        print(p)
        if "CO2" not in str(p):
            break_trajectory(p)
        # xyz_dir = p.parent / p.stem
        # highest_xyz = max([p for p in xyz_dir.rglob("*.xyz")])
        # adslab = p.parent.stem
        # print(adslab)
        # print(highest_xyz)
        # shutil.copy(
        #     highest_xyz,
        #     Path("..", "methanol_chemreasoner_results")
        #     / (adslab.replace("*", "") + ".xyz"),
        # )


# model weights have to placed in data/model_weights
