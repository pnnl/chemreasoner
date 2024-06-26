"""Calculate the reward for a set of structures from the microstructure planner."""

import json
import pickle
import random
import sys

from pathlib import Path

import numpy as np

from ase import Atoms
import ase.build as build
from ase.io import read, Trajectory

from ocdata.core import Adsorbate

sys.path.append("src")
from nnp.oc import OCAdsorptionCalculator
from nnp.uncertainty_prediction import UncertaintyCalculator
from structure_creation.digital_twin import CatalystDigitalTwin

with open(Path("data", "input_data", "oc", "oc_20_adsorbates.pkl"), "rb") as f:
    oc_20_ads_structures = pickle.load(f)
    oc_20_ads_structures = {
        v[1]: (v[0], v[2:]) for k, v in oc_20_ads_structures.items()
    }

with open(Path("data", "input_data", "oc") / "nist_adsorbates.pkl", "rb") as f:
    nist_ads_structures = pickle.load(f)


class AdsorptionEnergyCalculator:
    reference_energy_key = "e_slab"

    def __init__(
        self,
        atomistic_calc: OCAdsorptionCalculator,
        adsorbates_syms: list[str],
        num_augmentations_per_site: int = 1,
    ):
        """Initialize self, setting the data_dir."""
        self.data_dir = atomistic_calc.traj_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.num_augmentations_per_site = num_augmentations_per_site

        self.e_tot_dir = self.data_dir / "e_tot"
        self.e_tot_dir = self.data_dir / "e_slab"

        self.calc = atomistic_calc

        self.adsorbates_syms = adsorbates_syms

    def __call__(
        self,
        catalyst_structures: list[CatalystDigitalTwin],
        catalyst_names: list[str] = None,
    ):
        """Return the adsorption energy reward for the given structures."""
        if catalyst_names is None:
            catalyst_names = list(range(len(catalyst_structures)))

        e_tot_structures, e_tot_names = self.gather_total_energy_structures(
            catalyst_structures, catalyst_names
        )

        e_slab_structures, e_slab_names = (
            self.gather_slab_energy_structures(  # TODO: Don't re-calculate slab reference energies
                catalyst_structures, catalyst_names
            )
        )

        all_structures = e_tot_structures + e_slab_structures
        all_names = e_tot_names + e_slab_names

        # split into completed and incompleted calculations to avoid repeat work
        complete_names, complete_structures = [], []
        incomplete_names, incomplete_structures = [], []
        for structure, n in zip(all_structures, all_names):
            if self.check_complete(n):
                complete_names.append(n)
                complete_structures.append(self.fetch_complete_structure(n))
            else:
                incomplete_names.append(n)
                incomplete_structures.append(structure)

        relaxed_atoms = self.calc.batched_relax_atoms(
            atoms=incomplete_structures, atoms_names=incomplete_names
        )
        # Re-Combine complete/incomplete lists
        all_names = complete_names + incomplete_names
        all_structures = complete_structures + relaxed_atoms

        results = self._unpack_results(all_structures, all_names, catalyst_names)
        print(results)
        with open("test_gnn_results.json", "w") as f:
            json.dump(results, f)
        return results

    def get_relaxed_structures(
        self,
        catalyst_structures: list[CatalystDigitalTwin],
        catalyst_names: list[str] = None,
    ):
        """Return the adsorption energy reward for the given structures."""
        if catalyst_names is None:
            catalyst_names = list(range(len(catalyst_structures)))

        e_tot_structures, e_tot_names = self.gather_total_energy_structures(
            catalyst_structures, catalyst_names
        )

        e_slab_structures, e_slab_names = (
            self.gather_slab_energy_structures(  # TODO: Don't re-calculate slab reference energies
                catalyst_structures, catalyst_names
            )
        )

        all_structures = e_tot_structures + e_slab_structures
        all_names = e_tot_names + e_slab_names

        # split into completed and incompleted calculations to avoid repeat work
        complete_names, complete_structures = [], []
        incomplete_names, incomplete_structures = [], []
        for structure, n in zip(all_structures, all_names):
            if self.check_complete(n):
                complete_names.append(n)
                complete_structures.append(self.fetch_complete_structure(n))
            else:
                incomplete_names.append(n)
                incomplete_structures.append(structure)

        relaxed_atoms = self.calc.batched_relax_atoms(
            atoms=incomplete_structures, atoms_names=incomplete_names
        )
        # Re-Combine complete/incomplete lists
        all_names = complete_names + incomplete_names
        all_structures = complete_structures + relaxed_atoms

        results = self._unpack_results_structures(
            all_structures, all_names, catalyst_names
        )
        print(results)
        with open("test_gnn_results.json", "w") as f:
            json.dump(results, f)
        return results

    def check_complete(self, atoms_name):
        """Fetch the trajectory associated with the given atoms_names."""
        return (
            self.data_dir / (atoms_name + ".traj")
        ).exists()  # TODO: Put trajectories in db and change this code

    def fetch_complete_structure(self, atoms_name):
        """Fetch the trajectory associated with the given atoms_names."""
        return Trajectory(str(self.data_dir / (atoms_name + ".traj")))[
            -1
        ]  # TODO: Put trajectories in db and change this code

    def _unpack_results(self, relaxed_atoms, atoms_names, catalyst_names):
        """Unpack the results of the relaxation."""
        results = {}
        for catalyst_name in catalyst_names:
            indices = [
                i for i in range(len(atoms_names)) if catalyst_name in atoms_names[i]
            ]  # Should be fine for uuid catalyst names
            names = [atoms_names[i] for i in indices]
            structures = [relaxed_atoms[i] for i in indices]
            results[catalyst_name] = {}
            for name, structure in zip(names, structures):
                key = Path(name).stem.split("_")[-1]
                key = "e_slab" if key == "slab" else key  # e_slab key has _ in it
                value = (
                    structure.get_potential_energy()
                    if not isinstance(structure.get_potential_energy(), list)
                    else structure.get_potential_energy()[0]
                )

                results[catalyst_name].update({key: value})
        return results

    def _unpack_results_structures(self, relaxed_atoms, atoms_names, catalyst_names):
        """Unpack the results of the relaxation, returning the relaxed structures."""
        results = {}
        for catalyst_name in catalyst_names:
            indices = [
                i for i in range(len(atoms_names)) if catalyst_name in atoms_names[i]
            ]  # Should be fine for uuid catalyst names
            names = [atoms_names[i] for i in indices]
            structures = [relaxed_atoms[i] for i in indices]
            results[catalyst_name] = {}
            for name, structure in zip(names, structures):
                key = Path(name).stem.split("_")[-1]
                key = "e_slab" if key == "slab" else key  # e_slab key has _ in it
                value = structure

                results[catalyst_name].update({key: value})
        return results

    def gather_total_energy_structures(
        self, structures: list[CatalystDigitalTwin], names
    ):
        """Calculate the total energy for the given structures."""

        # Do total energy calculation
        e_tot_names = []
        e_tot_structures = []
        for n, struct in zip(names, structures):
            for ads_sym in self.adsorbates_syms:
                adsorbate_atoms = self.get_adsorbate_atoms(ads_sym)
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

    def gather_slab_energy_structures(
        self, structures: list[CatalystDigitalTwin], names
    ):
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

    @staticmethod
    def get_adsorbate_atoms(ads_syms: str):
        """Get the adsorbate atoms associated with the given adsorbates syms."""
        return ads_symbols_to_structure(ads_syms)

    def adsorbate_reference_energy(self, ads_syms: str):
        """Get the adsorbate reference energy from the given adsorbate syns."""
        ats = self.get_adsorbate_atoms(ads_syms=ads_syms)
        e_ref = 0
        for n in ats.get_atomic_numbers():
            e_ref += self.calc.ads_references[n]
        return e_ref


class AdsorptionEnergyUncertaintyCalculator:
    reference_energy_key = "e_slab"

    def __init__(
        self,
        uncertainty_calc: UncertaintyCalculator,
        adsorbates_syms: list[str],
    ):
        """Initialize self, setting the data_dir."""
        self.data_dir = uncertainty_calc.traj_dir
        self.e_tot_dir = self.data_dir / "e_tot"
        self.e_tot_dir = self.data_dir / "e_slab"

        self.calc = uncertainty_calc

        self.adsorbates_syms = adsorbates_syms

    def __call__(
        self,
        catalyst_structures: list[CatalystDigitalTwin],
        catalyst_names: list[str] = None,
    ):
        """Return the adsorption energy reward for the given structures."""
        all_structures, all_names = self.fetch_calculated_atoms(
            catalyst_structures, catalyst_names
        )

        uncertainty_values = self.calc.batched_uncertainty_calculation(
            atoms=all_structures, atoms_names=all_names
        )

        results = self._unpack_results(uncertainty_values, all_names, catalyst_names)
        with open("test_uq_results.json", "w") as f:
            json.dump(results, f)
        return results

    def check_complete(self, atoms_name):
        """Fetch the trajectory associated with the given atoms_names."""
        return (
            self.data_dir / (atoms_name + ".traj")
        ).exists()  # TODO: Put trajectories in db and change this code

    def fetch_calculated_atoms(
        self,
        catalyst_structures: list[CatalystDigitalTwin],
        catalyst_names: list[str] = None,
    ):
        """Fetch the structures associated with the given structures."""
        if catalyst_names is None:
            catalyst_names = list(range(len(catalyst_structures)))

        e_tot_structures, e_tot_names = self.gather_total_energy_structures(
            catalyst_structures, catalyst_names
        )

        e_slab_structures, e_slab_names = (
            self.gather_slab_energy_structures(  # TODO: Don't re-calculate slab reference energies
                catalyst_structures, catalyst_names
            )
        )

        all_structures = e_tot_structures + e_slab_structures
        all_names = e_tot_names + e_slab_names

        # split into completed and incompleted calculations to avoid repeat work
        all_structures = []
        for n in all_names:
            if not self.check_complete(n):
                raise FileNotFoundError(
                    f"The simulation had not completed for structure {n}"
                )
            all_structures.append(self.fetch_complete_structure(n))
        return all_structures, all_names

    def fetch_complete_structure(self, atoms_name):
        """Fetch the trajectory associated with the given atoms_names."""
        return Trajectory(str(self.data_dir / (atoms_name + ".traj")))[
            -1
        ]  # TODO: Put trajectories in db and change this code

    def _unpack_results(self, uncertainties, atoms_names, catalyst_names):
        """Unpack the results of the relaxation."""
        results = {}
        for catalyst_name in catalyst_names:
            indices = [
                i for i in range(len(atoms_names)) if catalyst_name in atoms_names[i]
            ]  # Should be fine for uuid catalyst names
            names = [atoms_names[i] for i in indices]
            these_uncertainties = [uncertainties[i] for i in indices]
            results[catalyst_name] = {}
            for name, uncertainty in zip(names, these_uncertainties):
                key = Path(name).stem.split("_")[-1]
                key = "e_slab" if key == "slab" else key  # e_slab key has _ in it
                value = uncertainty

                results[catalyst_name].update({key: value})
        return results

    def gather_total_energy_structures(
        self, structures: list[CatalystDigitalTwin], names
    ):
        """Calculate the total energy for the given structures."""

        # Do total energy calculation
        e_tot_names = []
        e_tot_structures = []
        for n, struct in zip(names, structures):
            for ads_sym in self.adsorbates_syms:
                e_tot_name = str(Path("trajectories_e_tot") / (n + f"_{ads_sym}"))
                (self.data_dir / e_tot_name).parent.mkdir(parents=True, exist_ok=True)
                e_tot_names.append(e_tot_name)
                e_tot_structures.append(self.fetch_complete_structure(e_tot_name))

        return e_tot_structures, e_tot_names

    def gather_slab_energy_structures(
        self, structures: list[CatalystDigitalTwin], names
    ):
        """Calculate the slabal energy for the given structures."""

        # Do slabal energy calculation
        e_slab_names = []
        e_slab_structures = []
        for n, struct in zip(names, structures):
            e_slab_name = str(Path("trajectories_e_slab") / (n + "_slab"))
            (self.data_dir / e_slab_name).parent.mkdir(parents=True, exist_ok=True)
            e_slab_names.append(e_slab_name)
            e_slab_structures.append(self.fetch_complete_structure(e_slab_name))

        return e_slab_structures, e_slab_names

    @staticmethod
    def get_adsorbate_atoms(ads_syms: str):
        """Get the adsorbate atoms associated with the given adsorbates syms."""
        return ads_symbols_to_structure(ads_syms)

    def adsorbate_reference_energy(self, ads_syms: str):
        """Get the adsorbate reference energy from the given adsorbate syns."""
        ats = self.get_adsorbate_atoms(ads_syms=ads_syms)
        e_ref = 0
        for n in ats.get_atomic_numbers():
            e_ref += self.calc.ads_references[n]
        return e_ref


def ads_symbols_to_structure(syms: str):
    """Turn adsorbate symbols to a list of strings."""
    if "*" in syms:
        ats = oc_20_ads_structures[syms][0].copy()
        ats.info.update({"binding_molecules": oc_20_ads_structures[syms][1][0].copy()})

    elif syms in map(lambda s: s.replace("*", ""), oc_20_ads_structures.keys()):
        idx = list(
            map(lambda s: s.replace("*", ""), oc_20_ads_structures.keys())
        ).index(syms)
        given_syms = syms
        syms = list(oc_20_ads_structures.keys())[idx]
        ats = oc_20_ads_structures[syms][0].copy()
        ats.info.update({"given_syms": given_syms})
        ats.info.update(
            {"binding_molecules": oc_20_ads_structures[syms][1][0].copy()}
        )  # get binding indices
    elif syms.lower() == "ethanol":
        return ads_symbols_to_structure("*OHCH2CH3")
    elif syms.lower() == "methanol":
        return ads_symbols_to_structure("*OHCH3")
    elif syms.lower() == "methyl":
        return ads_symbols_to_structure("*CH3")
    elif syms.lower() in nist_ads_structures.keys():
        return nist_ads_structures[syms.lower()]
    else:
        ats = build.molecule(syms)
    ats.info.update({"syms": syms})
    return ats


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


class TestStructure:
    def __init__(self, value=None):
        self.value = [random.random()] if value is None else [value]

    def get_potential_energy(self):
        """Return the value of self."""
        return self.value
