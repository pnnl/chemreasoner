"""Calculate the reward for a set of structures from the microstructure planner."""

import sys

from pathlib import Path

import numpy as np

from ase import Atoms
from ase.io import read

from ocdata.core import Adsorbate, AdsorbateSlabConfig

sys.path.append("src")
from llm import ase_interface
from nnp.oc import OCAdsorptionCalculator
from structure_creation.digital_twin import CatalystDigitalTwin


class AdsorptionEnergyCalculator:
    def __init__(
        self,
        atomistic_calc: OCAdsorptionCalculator,
        adsorbates_syms: list[str],
        num_augmentations_per_site: int = 1,
    ):
        """Initialize self, setting the data_dir."""
        self.data_dir = atomistic_calc.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.e_tot_dir = self.data_dir / "e_tot"
        self.e_tot_dir = self.data_dir / "e_slab"

        self.calc = atomistic_calc

        self.adsorbates_syms = adsorbates_syms

    def __call__(
        self,
        catalyst_structures: list[CatalystDigitalTwin],
        adsorbates_syms: list[str],
        catalyst_names: list[str] = None,
    ):
        """Return the adsorption energy reward for the given structures."""
        if catalyst_names is None:
            catalyst_names = list(range(len(catalyst_structures)))

        e_tot_structures, e_tot_names = self.gather_total_structures(
            catalyst_structures, adsorbates_syms, catalyst_names
        )

        e_slab_structures, e_slab_names = (
            self.gather_slab_structures(  # TODO: Don't re-calculate slab reference energies
                catalyst_structures, catalyst_names
            )
        )

        all_structures = e_tot_structures + e_slab_structures
        all_names = e_tot_names + e_slab_names

        relaxed_atoms = self.calc.batched_relax_atoms(
            atoms=all_structures, atoms_names=all_names
        )
        results = self._unpack_results(
            relaxed_atoms, catalyst_names, len_e_slab=len(e_slab_names)
        )
        return results

    def _unpack_results(self, relaxed_atoms, catalyst_names, len_e_slab):
        """Unpack the results of the relaxation."""
        e_tot_results = relaxed_atoms[:-len_e_slab]
        e_slab_results = relaxed_atoms[-len_e_slab:]
        results = {}
        for i, e_slab in enumerate(e_slab_results):
            catalyst_name = catalyst_names[i]
            results[catalyst_name] = {"e_slab": e_slab.get_potential_energy()}
            for j, ads_sym in enumerate(self.adsorbates_syms):
                e_tot = e_tot_results[i * len(self.adsorbates_syms) + j]
                results[catalyst_name].update(
                    {"adsorbate_syms": e_tot.get_potential_energy()}
                )
        return results

    def gather_total_energy(
        self, structures: list[CatalystDigitalTwin], adsorbates_syms: list[str], names
    ):
        """Calculate the total energy for the given structures."""

        # Do total energy calculation
        e_tot_names = []
        e_tot_structures = []
        for n, struct in zip(names, structures):
            for ads_sym in adsorbates_syms:
                adsorbate_atoms = ase_interface.ads_symbols_to_structure(ads_sym)
                binding_atoms = adsorbate_atoms.info.get("binding_sites", np.array([0]))
                adsorbate_object = Adsorbate(
                    adsorbate_atoms, adsorbate_binding_indices=binding_atoms
                )
                adslab_config = struct.return_adslab_config(
                    adsorbate=adsorbate_object,
                    num_augmentations_per_site=self.num_augmentations_per_site,
                )
                ats = adslab_config.atoms_list[0]
                ats.info.update(adslab_config.metadata_list[0])
                e_tot_name = str(Path("trajectories_e_tot") / (n + f"_{ads_sym}"))
                (self.data_dir / e_tot_name).parent.mkdir(parents=True, exist_ok=True)
                e_tot_names.append(e_tot_name)
                e_tot_structures.append(ats)

        return e_tot_structures, e_tot_names

    def gather_slab_energy(self, structures: list[CatalystDigitalTwin], names):
        """Calculate the slabal energy for the given structures."""

        # Do slabal energy calculation
        e_slab_names = []
        e_slab_structures = []
        for n, struct in zip(names, structures):
            slab_structure = struct.return_slab()
            e_slab_name = str(Path("trajectories_e_slab") / (n + "_slab"))
            (self.data_dir / e_slab_name).parent.mkdir(parents=True, exist_ok=True)
            e_slab_names.append(e_slab_name)
            e_slab_structures.append(slab_structure)

        return e_slab_structures, e_slab_names


if __name__ == "__main__":
    calc = OCAdsorptionCalculator(
        **{
            "model": "gemnet-oc-22",
            "traj_dir": "test_trajs",
            "batch_size": 32,
            "device": "cuda",
            "ads_tag": 2,
            "fmax": 0.03,
            "steps": 3,
        }
    )
