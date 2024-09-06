"""Calculate the reward for a set of structures from the microstructure planner."""

import json
import logging
import pickle
import random
import sys

from pathlib import Path

import numpy as np

from ase import Atoms
import ase.build as build
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory, write
from ase.io.trajectory import TrajectoryWriter

from ocdata.core import Adsorbate
from ocdata.utils.flag_anomaly import DetectTrajAnomaly

sys.path.append("src")
from nnp.oc import OCAdsorptionCalculator
from nnp.uncertainty_prediction import UncertaintyCalculator
from structure_creation.digital_twin import CatalystDigitalTwin

logging.getLogger().setLevel(logging.INFO)

with open(Path("data", "input_data", "oc", "oc_20_adsorbates.pkl"), "rb") as f:
    oc_20_ads_structures = pickle.load(f)
    oc_20_ads_structures = {
        v[1]: (v[0], v[2:]) for k, v in oc_20_ads_structures.items()
    }

with open(Path("data", "input_data", "reactions", "co2_to_methanol.pkl"), "rb") as f:
    co2_to_methanol_structures = pickle.load(f)

with open(Path("data", "input_data", "reactions", "co_to_ethanol.pkl"), "rb") as f:
    co_to_ethanol_structures = pickle.load(f)

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
        self.e_slab_dir = self.data_dir / "e_slab"

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
        logging.info("Checking for completed relaxations...")
        complete_names, complete_structures = [], []
        incomplete_names, incomplete_structures = [], []
        for structure, n in zip(all_structures, all_names):
            if self.check_complete(n):
                complete_names.append(n)
                complete_structures.append(self.fetch_complete_structure(n))
            else:
                incomplete_names.append(n)
                incomplete_structures.append(structure)
        logging.info(
            f"#complete: {len(complete_names)}\t#incomplete: {len(incomplete_names)}"
        )
        if len(incomplete_structures) > 0:
            relaxed_atoms = self.calc.batched_relax_atoms(
                atoms=incomplete_structures, atoms_names=incomplete_names
            )
        else:
            relaxed_atoms = []

        # Check the relaxed structures
        for i in range(len(incomplete_structures)):
            tmp_ats = relaxed_atoms[i]
            relaxed_atoms[i] = self.fetch_complete_structure(incomplete_names[i])
            logging.info(f"{len(relaxed_atoms[i])} in traj {len(tmp_ats)} in output")

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
                incom_structure = self.fetch_incomplete_structure(n)
                incomplete_structures.append(
                    structure if incom_structure is None else incom_structure
                )

        if len(incomplete_structures) > 0:
            relaxed_atoms = self.calc.batched_relax_atoms(
                atoms=incomplete_structures, atoms_names=incomplete_names
            )
            for atoms, name in zip(relaxed_atoms, incomplete_names):
                self.save_complete_structure(atoms, name)
        else:
            relaxed_atoms = []

        # Check the relaxed structures
        for i in range(len(incomplete_structures)):
            init_struct = incomplete_structures[i]
            final_struct = relaxed_atoms[i]

            good_structure = self.check_structure(init_struct, final_struct)
            if not good_structure:
                relaxed_atoms[i] = self.nan_energy(relaxed_atoms[i])

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
        # TODO: Put trajectories in db and change this code
        if (self.data_dir / (atoms_name + ".traj")).exists():
            traj = Trajectory(str(self.data_dir / (atoms_name + ".traj")))
            if len(traj) == 0:
                print("not started")
                return False

            try:
                ats = traj[-1]
                fmax = np.max(np.sqrt(np.sum(ats.get_forces() ** 2, axis=1)))
                steps = len(traj)
                if fmax <= self.calc.fmax or steps >= self.calc.steps:
                    return True
                else:
                    return False
            except Exception as err:
                logging.warning(
                    f"Could not read file {self.data_dir / (atoms_name + '.traj')} with error {err}."
                )
                return False

        else:
            return False

    def fetch_complete_structure(self, atoms_name):
        """Fetch the trajectory associated with the given atoms_names."""
        # TODO: Put trajectories in db and change this code
        traj = Trajectory(str(self.data_dir / (atoms_name + ".traj")))
        good_structure = self.check_structure(traj[0], traj[-1])
        if not good_structure:
            return self.nan_energy(traj[-1])
        else:
            return traj[-1]

    def fetch_incomplete_structure(self, atoms_name):
        """Fetch the incomplete structure for name."""
        try:
            return Trajectory(str(self.data_dir / (atoms_name + ".traj")))[0]
        except FileNotFoundError:
            return None

    def check_structure(self, initial_structure: Atoms, final_structure: Atoms):
        """Check the given structure for good convergence, using criteria from OpenCatalyst Project."""

        error_code = self.get_convergence_error_code(initial_structure, final_structure)
        return error_code == 0 or error_code == 6

    def get_convergence_error_code(self, initial_structure, final_structure):
        """Check the given structure for convergence error code, using criteria from OpenCatalyst Project."""
        if not all(
            [
                t1 == t2
                for t1, t2 in zip(
                    initial_structure.get_tags(), final_structure.get_tags()
                )
            ]
        ):
            write("init_test.xyz", initial_structure)
            write("final_test.xyz", final_structure)
        anomaly_detector = DetectTrajAnomaly(
            init_atoms=initial_structure,
            final_atoms=final_structure,
            atoms_tag=initial_structure.get_tags(),
        )
        fmax = np.max(np.sqrt(np.sum(final_structure.get_forces() ** 2, axis=1)))
        if anomaly_detector.has_surface_changed():
            return 3
        elif 2 in initial_structure.get_tags():
            if anomaly_detector.is_adsorbate_dissociated():
                return 1
            elif anomaly_detector.is_adsorbate_desorbed():
                return 2
            elif anomaly_detector.is_adsorbate_intercalated():
                return 5
            # No value for 4. This was used for incorrect CHCOH placement in OCP dataset and is not measured here
            elif fmax > self.calc.fmax:
                return 6
            else:
                return 0
        else:
            return 0

    def fetch_error_code(self, atoms_name):
        """Fetch the trajectory associated with the given atoms_names."""
        # TODO: Put trajectories in db and change this code
        traj = Trajectory(str(self.data_dir / (atoms_name + ".traj")))
        return self.get_convergence_error_code(traj[0], traj[-1])

    def get_error_codes_name(self, catalyst_name: str):
        """Check the given structure for convergence error code, using criteria from OpenCatalyst Project."""
        codes = {}
        for ads in self.adsorbates_syms:
            codes[ads] = self.fetch_error_code(f"{catalyst_name}_{ads}")
        codes[self.reference_energy_key] = self.fetch_error_code(
            f"{catalyst_name}_{self.reference_energy_key}"
        )
        return codes

    def get_error_codes_catalysts(self, catalyst_names: list[str]):
        """Check the given structure for convergence error code, using criteria from OpenCatalyst Project."""
        e_tot_names = self.gather_total_energy_names(catalyst_names)

        e_slab_names = self.gather_slab_energy_names(  # TODO: Don't re-calculate slab reference energies
            catalyst_names
        )
        all_names = e_tot_names + e_slab_names

        codes = {}
        for name in all_names:
            uuid = Path(name).stem.split("_")[0]
            key = Path(name).stem.replace(f"{uuid}_", "")
            if uuid not in codes:
                codes[uuid] = {}
            codes[uuid][key] = self.fetch_error_code(name)
        return codes

    def nan_energy(self, structure: Atoms) -> Atoms:
        """Return copy of given structure with potential_energy of nan."""
        ats = structure.copy()
        res = structure.calc.results
        res["energy"] = np.nan if not isinstance(res["energy"], list) else [np.nan]
        res["forces"] = structure.get_forces() * np.nan
        ats.calc = SinglePointCalculator(structure, **res)
        return ats

    def save_complete_structure(self, structure, atoms_name):
        """Fetch the trajectory associated with the given atoms_names."""
        fmax = np.max(np.sqrt(np.sum(structure.get_forces() ** 2, axis=1)))
        if fmax <= self.calc.fmax:
            # Only need to save if forces are converged, not steps
            traj_writer = TrajectoryWriter(
                str(self.data_dir / (atoms_name + ".traj")), mode="a"
            )
            traj_writer.write(structure)
            # TODO: Put trajectories in db and change this code

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
                key = (
                    self.reference_energy_key if key == "slab" else key
                )  # e_slab key has _ in it
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
                key = (
                    self.reference_energy_key if key == "slab" else key
                )  # e_slab key has _ in it
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
                )  # TODO: Fix this to allow augmentations
                ats = adslab_config.atoms_list[0]
                ats.info.update(adslab_config.metadata_list[0])
                e_tot_name = str(Path("trajectories_e_tot") / (n + f"_{ads_sym}"))
                (self.data_dir / e_tot_name).parent.mkdir(parents=True, exist_ok=True)
                e_tot_names.append(e_tot_name)
                e_tot_structures.append(ats)

        return e_tot_structures, e_tot_names

    def gather_total_energy_names(self, names: list[str]):
        """Calculate the total energy for the given structures."""

        # Do total energy calculation
        e_tot_names = []
        for n in names:
            for ads_sym in self.adsorbates_syms:
                e_tot_name = str(Path("trajectories_e_tot") / (n + f"_{ads_sym}"))
                (self.data_dir / e_tot_name).parent.mkdir(parents=True, exist_ok=True)
                e_tot_names.append(e_tot_name)

        return e_tot_names

    def gather_slab_energy_structures(
        self, structures: list[CatalystDigitalTwin], names
    ):
        """Calculate the slabal energy for the given structures."""
        # Do slabal energy calculation
        e_slab_names = []
        e_slab_structures = []
        for n, struct in zip(names, structures):
            slab_structure = struct.return_slab()
            e_slab_name = str(
                Path(f"trajectories_{self.reference_energy_key}")
                / (f"{n}_{self.reference_energy_key}")
            )
            (self.data_dir / e_slab_name).parent.mkdir(parents=True, exist_ok=True)
            e_slab_names.append(e_slab_name)
            e_slab_structures.append(slab_structure)

        return e_slab_structures, e_slab_names

    def gather_slab_energy_names(self, names):
        """Calculate the slabal energy for the given structures."""

        # Do slabal energy calculation
        e_slab_names = []
        for n in names:
            e_slab_name = str(
                Path(f"trajectories_{self.reference_energy_key}")
                / (f"{n}_{self.reference_energy_key}")
            )
            (self.data_dir / e_slab_name).parent.mkdir(parents=True, exist_ok=True)
            e_slab_names.append(e_slab_name)

        return e_slab_names

    @staticmethod
    def get_adsorbate_atoms(ads_syms: str):
        """Get the adsorbate atoms associated with the given adsorbates syms."""
        return ads_symbols_to_structure(ads_syms)

    def adsorbate_reference_energy(self, ads_syms: str):
        """Get the adsorbate reference energy from the given adsorbate syms."""
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
        self.e_tot_dir = self.data_dir / self.reference_energy_key

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
                key = (
                    self.reference_energy_key if key == "slab" else key
                )  # e_slab key has _ in it
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
            e_slab_name = str(
                Path(f"trajectories_{self.reference_energy_key}")
                / (f"{n}_{self.reference_energy_key}")
            )
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
    if "*" in syms and syms in oc_20_ads_structures.keys():
        ats = oc_20_ads_structures[syms][0].copy()
        ats.info.update({"binding_sites": oc_20_ads_structures[syms][1][0].copy()})

    elif syms in map(lambda s: s.replace("*", ""), oc_20_ads_structures.keys()):
        idx = list(
            map(lambda s: s.replace("*", ""), oc_20_ads_structures.keys())
        ).index(syms)
        given_syms = syms
        syms = list(oc_20_ads_structures.keys())[idx]
        ats = oc_20_ads_structures[syms][0].copy()
        ats.info.update({"given_syms": given_syms})
        ats.info.update(
            {"binding_sites": oc_20_ads_structures[syms][1][0].copy()}
        )  # get binding indices
    elif syms in co2_to_methanol_structures.keys():
        ats, binding_sites = co2_to_methanol_structures[syms]
        ats = ats.copy()
        ats.info.update({"binding_sites": binding_sites.copy()})
    elif syms in co_to_ethanol_structures.keys():
        ats, binding_sites = co_to_ethanol_structures[syms]
        ats = ats.copy()
        ats.info.update({"binding_sites": binding_sites.copy()})
    elif syms.lower() == "ethanol":
        return ads_symbols_to_structure("*OHCH2CH3")
    elif syms.lower() == "methanol":
        return ads_symbols_to_structure("*OHCH3")
    elif syms.lower() == "methyl":
        return ads_symbols_to_structure("*CH3")
    elif syms.lower() in nist_ads_structures.keys():
        return nist_ads_structures[syms.lower()]
    elif syms == "*CH*O":
        ats = ads_symbols_to_structure("*CHO")
        ats.info.update({"syms": "*CH*O"})
        ats.info.update({"binding_sites": np.array([0, 2])})
        return ats
    else:
        ats = build.molecule(syms)
    ats.info.update({"syms": syms})
    return ats


class TestStructure:
    def __init__(self, value=None):
        self.value = [random.random()] if value is None else [value]

    def get_potential_energy(self):
        """Return the value of self."""
        return self.value


if __name__ == "__main__":
    print(oc_20_ads_structures["*OCH2"])
    print(ads_symbols_to_structure("*CHO").info)
    print(ads_symbols_to_structure("*COOH").info)
