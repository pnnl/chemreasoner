"""Functions for reward with microstructure planner."""

import logging
import sys
import time

from copy import deepcopy

from ase import Atoms
from ase.units import kB
import numpy as np

sys.path.append("src")
from nnp.oc import OCAdsorptionCalculator
from nnp.uncertainty_prediction import UncertaintyCalculator
from structure_creation.digital_twin import CatalystDigitalTwin
from search.reward.adsorption_energy_reward import (
    AdsorptionEnergyCalculator,
    AdsorptionEnergyUncertaintyCalculator,
)

logging.getLogger().setLevel(logging.INFO)


class MicrostructureRewardFunction:

    def __init__(
        self,
        reaction_pathways: list[list[str]],
        calc: OCAdsorptionCalculator,
        num_augmentations_per_site: int = 1,
        T: float = None,
    ):
        """Return self, with the given reaction_pathways and calculator initialized."""
        self._cached_calculations = {}
        # Check to make sure reaction pathways specifies number of intermediates, else convert
        if isinstance(reaction_pathways[0][0], dict):
            self.reaction_pathways = reaction_pathways
        else:
            self.reaction_pathways = [
                [{syms: 1} for syms in pathway] for pathway in reaction_pathways
            ]
        self._all_adsorbate_symbols = list(
            {
                ads_sym
                for ads_list in self.reaction_pathways
                for ads_dict in ads_list
                for ads_sym in ads_dict.keys()
            }
        )
        self.calc = calc
        self.num_augmentations_per_site = num_augmentations_per_site

        self.ads_e_calc = AdsorptionEnergyCalculator(
            atomistic_calc=self.calc,
            adsorbates_syms=self._all_adsorbate_symbols,
            num_augmentations_per_site=self.num_augmentations_per_site,
        )
        self.T = T if T is not None else 300

    def __call__(self, structures: list[CatalystDigitalTwin]):
        """Call the reward values for the given list of structures."""
        energies = self.ads_e_calc(
            catalyst_structures=structures, catalyst_names=[s._id for s in structures]
        )
        self._cached_calculations.update(energies)
        final_values = self._calculate_final_reward(energies=energies)
        return [final_values[s._id] for s in structures]

    def _calculate_final_reward(self, energies: dict[str, float]):
        """Calculate the final reward associated with the given energies."""
        reactant_energies = self._parse_reactant_energies(energies)
        energy_barriers = self._parse_energy_barriers(energies)
        rewards = {  # TODO: Do a better calculation for these
            k: np.exp(
                -1 * (reactant_energies[k] + energy_barriers[k]["best"]) / kB / self.T
            )
            for k in reactant_energies.keys()
        }

        return rewards

    def fetch_total_energy_results(self, structures: list[CatalystDigitalTwin]):
        """Fetch the energies associated with the given structures."""
        results = {}
        for s in structures:
            row = self._cached_calculations[s._id]
            results[s._id] = {ads_key: e_tot for ads_key, e_tot in row.items()}
        return deepcopy(results)

    def fetch_adsorption_energy_results(self, structures: list[CatalystDigitalTwin]):
        """Fetch the energies associated with the given structures."""
        results = {}
        for s in structures:
            row = self._cached_calculations[s._id]
            results[s._id] = {
                ads_key: (
                    e_tot
                    - row[self.ads_e_calc.reference_energy_key]
                    - self.ads_e_calc.adsorbate_reference_energy(ads_key)
                    if ads_key != self.ads_e_calc.reference_energy_key
                    else e_tot
                )
                for ads_key, e_tot in row.items()
            }
        return deepcopy(results)

    def fetch_error_codes(self, structures: list[CatalystDigitalTwin]):
        """Fetch the energies associated with the given structures."""
        results = self.ads_e_calc.get_error_codes_catalysts([s._id for s in structures])
        return deepcopy(results)

    def fetch_reward_results(self, structures: list[CatalystDigitalTwin]):
        """Fetch the rewards associated with the given structures."""
        energies = self.fetch_total_energy_results(structures)
        reactant_energies = self._parse_reactant_energies(energies)
        energy_barriers = self._parse_energy_barriers(energies)
        return {
            s._id: {
                "reward_function_1": reactant_energies[s._id],
                "reward_function_2": energy_barriers[s._id]["best"],
                "reward": np.exp(
                    -1
                    * (reactant_energies[s._id] + energy_barriers[s._id]["best"])
                    / kB
                    / self.T
                ),  # TODO: Do a better calculation for these
            }
            for s in structures
        }

    def _parse_reactant_energies(self, energy_results: dict[str, dict[str, float]]):
        """Parse the energies of the reactants for the reaction pathways."""
        symbols = list(
            {k for p in self.reaction_pathways for k in p[0].keys() if "C" in k}
        )
        if len(symbols) > 1:
            logging.warning(f"Length of reactant symbols is {len(symbols)}, not 1.")
        syms = symbols[0]
        energies = {
            catalyst: catalyst_results[syms]
            - catalyst_results[self.ads_e_calc.reference_energy_key]
            - self.ads_e_calc.adsorbate_reference_energy(syms)
            for catalyst, catalyst_results in energy_results.items()
        }
        return energies

    def _parse_energy_barriers(self, energy_results: dict[str, dict[str, float]]):
        """Parse the reaction barriers for the reaction pathways."""
        barriers = {}
        for catalyst, catalyst_results in energy_results.items():
            barriers[catalyst] = {}
            for i, pathway in enumerate(self.reaction_pathways):
                e = [
                    sum(
                        [
                            count
                            * (
                                catalyst_results[syms]
                                - self.ads_e_calc.adsorbate_reference_energy(syms)
                                - catalyst_results[self.ads_e_calc.reference_energy_key]
                            )
                            for syms, count in step.items()
                        ]
                    )
                    for step in pathway
                ]
                diffs = np.diff(e).tolist()
                barriers[catalyst].update({f"pathway_{i}": max(diffs)})
            barriers[catalyst].update({"best": min(barriers[catalyst].values())})

        return barriers


class MicrostructureUncertaintyFunction:

    def __init__(
        self,
        reaction_pathways: list[list[str]],
        calc: UncertaintyCalculator,
    ):
        """Return self, with the given reaction_pathways and calculator initialized."""
        self._cached_calculations = {}
        if isinstance(reaction_pathways[0][0], dict):
            self.reaction_pathways = reaction_pathways
        else:
            self.reaction_pathways = [
                [{syms: 1} for syms in pathway] for pathway in reaction_pathways
            ]
        self._all_adsorbate_symbols = list(
            {
                ads_sym
                for ads_list in self.reaction_pathways
                for ads_dict in ads_list
                for ads_sym in ads_dict.keys()
            }
        )
        self.calc = calc

        self.ads_e_calc = AdsorptionEnergyUncertaintyCalculator(
            uncertainty_calc=self.calc,
            adsorbates_syms=self._all_adsorbate_symbols,
        )

    def __call__(self, structures: list[CatalystDigitalTwin]):
        """Call the reward values for the given list of structures."""
        uncertainty_dictionary = self.ads_e_calc(
            catalyst_structures=structures, catalyst_names=[s._id for s in structures]
        )
        self._cached_calculations.update(uncertainty_dictionary)
        uncertainty_arrays = {
            k: np.sqrt(np.sum(np.array(list(v.values())) ** 2))
            for k, v in uncertainty_dictionary.items()
        }
        return [uncertainty_arrays[s._id] for s in structures]

    def fetch_calculated_atoms(self, structures: list[CatalystDigitalTwin]) -> Atoms:
        """Fetch the atoms associated with the given structures, filter by top_p uncertainty."""
        all_structures, all_names = self.ads_e_calc.fetch_calculated_atoms(
            catalyst_structures=structures, catalyst_names=[s._id for s in structures]
        )
        return all_structures, all_names

    # def _parse_reactant_energies(self, energy_results: dict[str, dict[str, float]]):
    #     """Parse the energies of the reactants for the reaction pathways."""
    #     symbols = list({p[0] for p in self.reaction_pathways})
    #     if len(symbols) > 1:
    #         logging.warning(f"Length of reactant symbols is {len(symbols)}, not 1.")
    #     syms = symbols[0]
    #     energies = {
    #         catalyst: catalyst_results[syms]
    #         - catalyst_results[self.ads_e_calc.reference_energy_key]
    #         - self.ads_e_calc.adsorbate_reference_energy(syms)
    #         for catalyst, catalyst_results in energy_results.items()
    #     }
    #     return energies

    def fetch_uncertainty_results(self, structures: list[CatalystDigitalTwin]):
        """Fetch the uncertainty results from the given structures."""
        results = {}
        for s in structures:
            row = self._cached_calculations[s._id]
            results[s._id] = row  # TODO: Is there aggregation to do?
        return deepcopy(results)

    # def _parse_energy_barriers(self, energy_results: dict[str, dict[str, float]]):
    #     """Parse the reaction barriers for the reaction pathways."""
    #     barriers = {}
    #     for catalyst, catalyst_results in energy_results.items():
    #         barriers[catalyst] = {}
    #         for i, pathway in enumerate(self.reaction_pathways):
    #             e = [
    #                 catalyst_results[syms]
    #                 - self.ads_e_calc.adsorbate_reference_energy(syms)
    #                 for syms in pathway
    #             ]
    #             diffs = np.diff(e).tolist()
    #             barriers[catalyst].update({f"pathway_{i}": max(diffs)})
    #         barriers[catalyst].update({"best": min(barriers[catalyst].values())})

    #     return barriers


if __name__ == "__main__":
    calc = OCAdsorptionCalculator(
        **{
            "model": "gemnet-oc-22",
            "traj_dir": "test_trajs",
            "batch_size": 32,
            "device": "cuda",
            "ads_tag": 2,
            "fmax": 0.03,
            "steps": 200,
        }
    )
    start = time.time()

    end = time.time()
    print(f"TIME: {end-start}")
