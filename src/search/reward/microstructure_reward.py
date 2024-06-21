"""Functions for reward with microstructure planner."""

import logging
import sys
import time

import numpy as np

from ase import Atoms

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
    ):
        """Return self, with the given reaction_pathways and calculator initialized."""
        self.reaction_pathways = reaction_pathways
        self._all_adsorbate_symbols = list(
            {ads_sym for ads_list in self.reaction_pathways for ads_sym in ads_list}
        )
        self.calc = calc
        self.num_augmentations_per_site = num_augmentations_per_site

        self.ads_e_reward = AdsorptionEnergyCalculator(
            atomistic_calc=self.calc,
            adsorbates_syms=self._all_adsorbate_symbols,
            num_augmentations_per_site=self.num_augmentations_per_site,
        )

    def __call__(self, structures: list[CatalystDigitalTwin]):
        """Call the reward values for the given list of structures."""
        energies = self.ads_e_reward(
            catalyst_structures=structures, catalyst_names=[s._id for s in structures]
        )
        reactant_energies = self._parse_reactant_energies(energies)
        energy_barriers = self._parse_energy_barriers(energies)
        final_values = {  # TODO: Do a better calculation for these
            k: -1 * (reactant_energies[k] / energy_barriers[k]["best"])
            for k in reactant_energies.keys()
        }
        return [final_values[s._id] for s in structures]

    def _parse_reactant_energies(self, energy_results: dict[str, dict[str, float]]):
        """Parse the energies of the reactants for the reaction pathways."""
        symbols = list({p[0] for p in self.reaction_pathways})
        if len(symbols) > 1:
            logging.warning(f"Length of reactant symbols is {len(symbols)}, not 1.")
        syms = symbols[0]
        energies = {
            catalyst: catalyst_results[syms]
            - catalyst_results[self.ads_e_reward.reference_energy_key]
            - self.ads_e_reward.adsorbate_reference_energy(syms)
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
                    catalyst_results[syms]
                    - self.ads_e_reward.adsorbate_reference_energy(syms)
                    for syms in pathway
                ]
                diffs = np.diff(e).tolist()
                barriers[catalyst].update({f"pathway_{i}": max(diffs)})
            barriers[catalyst].update({"best": min(barriers[catalyst].values())})

        return barriers


class MicrostructureUncertaintyFunction:

    _cached_uncertainties = {}

    def __init__(
        self,
        reaction_pathways: list[list[str]],
        calc: UncertaintyCalculator,
    ):
        """Return self, with the given reaction_pathways and calculator initialized."""
        self.reaction_pathways = reaction_pathways
        self._all_adsorbate_symbols = list(
            {ads_sym for ads_list in self.reaction_pathways for ads_sym in ads_list}
        )
        self.calc = calc

        self.ads_e_reward = AdsorptionEnergyUncertaintyCalculator(
            atomistic_calc=self.calc,
            adsorbates_syms=self._all_adsorbate_symbols,
        )

    def __call__(self, structures: list[CatalystDigitalTwin]):
        """Call the reward values for the given list of structures."""
        uncertainty_dictionary = self.ads_e_reward(
            catalyst_structures=structures, catalyst_names=[s._id for s in structures]
        )
        uncertainty_arrays = {
            k: np.sqrt(np.sum(np.array(list(v.values())) ** 2))
            for k, v in uncertainty_dictionary.items()
        }
        return [uncertainty_arrays[s._id] for s in structures]

    @classmethod
    def store_uncertainty(cls, uncertainty_dictionary):
        """Store the given uncertainty dictionary in class variable."""
        cls._cached_uncertainties.update({uncertainty_dictionary})

    @classmethod
    def get_stored_uncertainties(cls):
        """Get the uncertainties stored in class."""
        return cls._cached_uncertainties

    def fetch_calculated_atoms(self, structures) -> Atoms:
        """Fetch the atoms associated with the given structures, filter by top_p uncertainty."""
        all_structures, all_names = self.ads_e_reward.fetch_calculated_atoms(
            catalyst_structures=structures, catalyst_names=[s._id for s in structures]
        )

    def _parse_reactant_energies(self, energy_results: dict[str, dict[str, float]]):
        """Parse the energies of the reactants for the reaction pathways."""
        symbols = list({p[0] for p in self.reaction_pathways})
        if len(symbols) > 1:
            logging.warning(f"Length of reactant symbols is {len(symbols)}, not 1.")
        syms = symbols[0]
        energies = {
            catalyst: catalyst_results[syms]
            - catalyst_results[self.ads_e_reward.reference_energy_key]
            - self.ads_e_reward.adsorbate_reference_energy(syms)
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
                    catalyst_results[syms]
                    - self.ads_e_reward.adsorbate_reference_energy(syms)
                    for syms in pathway
                ]
                diffs = np.diff(e).tolist()
                barriers[catalyst].update({f"pathway_{i}": max(diffs)})
            barriers[catalyst].update({"best": min(barriers[catalyst].values())})

        return barriers


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
