"""Scripts for running atomic optimizations with ocp-models.

Must intsall the sub-module ocpmodels included in ext/ocp.
"""
import json
import pickle
import time
import wget
import yaml

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import Trajectory, write
from ase.neighborlist import build_neighbor_list
from ase.optimize import BFGS

from ocpmodels.common.relaxation.ase_utils import OCPCalculator, batch_to_atoms
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.typing import assert_is_instance
from ocpmodels.datasets.lmdb_dataset import data_list_collater
from ocpmodels.preprocessing.atoms_to_graphs import AtomsToGraphs


import torch
from torch_geometric.data import Batch

from .base_nnp import BaseAdsorptionCalculator


class OCAdsorptionCalculator(BaseAdsorptionCalculator):
    """Class to calculate adsorption energies. Follows Open Catalyst Porject methods."""

    model_weights_paths = Path("data", "model_weights")
    model_weights_paths.mkdir(parents=True, exist_ok=True)
    model_configs_paths = Path("ext", "ocp", "configs", "s2ef", "all")

    # (8/18/2023) reference values from:
    # https://arxiv.org/abs/2010.09990
    ads_references = {
        1: -3.477,
        6: -7.282,
        7: -8.083,
        8: -7.204,
    }

    def __init__(
        self,
        model: str,
        traj_dir: Path,
        batch_size=40,
        fmax=0.005,
        steps=150,
        device="cuda:0",
        adsorbed_structure_checker=None,
    ):
        """Create object from model class (gemnet or equiformer).

        Downloads weights if they are not available.
        """
        self.gnn_calls = 0
        self.gnn_time = 0
        self.device = device
        self.batch_size = batch_size
        self.fmax = fmax
        self.steps = steps
        self.model = model
        if self.model == "gemnet":
            self.model_path = self.model_weights_paths / "gemnet_t_direct_h512_all.pt"
            if not self.model_path.exists():
                print("Downloading weights for gemnet...")
                wget.download(
                    "https://dl.fbaipublicfiles.com/opencatalystproject/models/"
                    "2021_08/s2ef/gemnet_t_direct_h512_all.pt",
                    out=str(self.model_weights_paths),
                )
                print("Done!")
            self.config_path = self.model_configs_paths / "gemnet" / "gemnet-dT.yml"

        elif self.model == "equiformer":
            self.model_path = self.model_weights_paths / "eq2_153M_ec4_allmd.pt"
            if not self.model_path.exists():
                print("Downloading weights for equiformer...")
                wget.download(
                    "https://dl.fbaipublicfiles.com/opencatalystproject/models/"
                    "2023_06/oc20/s2ef/eq2_153M_ec4_allmd.pt",
                    out=str(self.model_weights_paths),
                )
                print("Done!")
            self.config_path = (
                self.model_configs_paths
                / "equiformer_v2"
                / "equiformer_v2_N@20_L@6_M@3_153M.yml"
            )

        else:
            raise ValueError(f"Unkown model {self.model}.")
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        if adsorbed_structure_checker is None:
            self.adsorbed_structure_checker = AdsorbedStructureChecker()

        self.traj_dir = traj_dir
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        # set up atoms to graphs object
        self.ats_to_graphs = AtomsToGraphs(
            r_edges=False,
            r_fixed=True,
            r_pbc=True,
        )

        self.ase_calc = None
        self.torch_calc = None

    @property
    def get_ase_calculator(self):
        """Return an ase calculator for self.

        Specifying device overrides self.device.
        """
        if self.ase_calc is None:
            # set up calculator for ase relaxations
            self.ase_calc = OCPCalculator(
                config_yml=str(self.config_path),
                checkpoint_path=str(self.model_path),
                cpu=self.device == "cpu",
            )
        return self.ase_calc

    @property
    def get_torch_model(self):
        """Return an ase calculator for self.

        Specifying device overrides self.device.
        """
        # set up calculator for ase relaxations
        if self.torch_calc is None:
            ase_calc = self.get_ase_calculator
            self.torch_calc = ase_calc.trainer
        return self.torch_calc

    def relax_atoms_ase(
        self,
        atoms: Atoms,
        device=None,
        fname=None,
        fmax=None,
        steps=None,
        **bfgs_kwargs,
    ):
        """Relax the postitions of the given atoms.

        Setting device overrides self.device
        """
        fmax = fmax if fmax is not None else self.fmax
        steps = steps if steps is not None else self.steps
        atoms = atoms.copy()
        self.prepare_atoms(atoms)

        atoms.calc = self.get_ase_calculator
        opt = BFGS(
            atoms, trajectory=self.traj_dir / fname if fname is not None else None
        )

        opt.run(fmax=fmax, steps=steps)
        return atoms

    def batched_relax_atoms(
        self,
        atoms: list[Atoms],
        atoms_names,
        device=None,
        batch_size=None,
        fmax=None,
        steps=None,
        **bfgs_kwargs,
    ):
        """Relax the postitions of the given atoms. Setting device overrides self."""
        atoms = self.copy_atoms_list(atoms)
        fmax = fmax if fmax is not None else self.fmax
        steps = steps if steps is not None else self.steps
        # Set up calculation for oc
        self.prepare_atoms_list(atoms)
        # convert to torch geometric batch
        batch = Batch.from_data_list(self.ats_to_graphs.convert_all(atoms))
        batch.sid = atoms_names
        batch = batch.to(device if device is not None else self.device)

        trainer = self.get_torch_model

        try:
            relax_opt = self.config["task"]["relax_opt"]
        except KeyError:
            relax_opt = {"memory": steps}  # only need to set memory and traj_dir

        relax_opt["traj_dir"] = self.traj_dir
        # assume 100 steps every time
        start = time.time()
        final_batch = ml_relax(
            batch=[batch],  # ml_relax always uses batch[0]
            model=trainer,
            steps=steps,
            fmax=fmax,
            relax_opt=relax_opt,
            save_full_traj=True,
            device=trainer.device,
        )
        end = time.time()
        self.gnn_calls += 100
        self.gnn_time += end - start

        final_atoms = batch_to_atoms(final_batch)
        return final_atoms

    def batched_adsorption_calculation(
        self,
        atoms: list[Atoms],
        atoms_names,
        device=None,
        **bfgs_kwargs,
    ):
        """Calculate adsorption energies from relaxed atomic positions."""
        adslabs = atoms
        atoms = self.copy_atoms_list(atoms)
        # Set up calculation for oc
        bulk_atoms = []
        ads_e = []
        for ats in atoms:
            bulk_ats = Atoms()
            e_ref = 0
            for i, t in enumerate(ats.get_tags()):
                if t == 0:  # part of the adsorbate
                    e_ref += self.ads_references[ats.get_atomic_numbers()[i]]
                else:  # part of the bulk
                    bulk_ats.append(ats[i])
            ads_e.append(e_ref)
            bulk_atoms.append(bulk_ats.copy())
        # convert to torch geometric batch
        batch = Batch.from_data_list(self.ats_to_graphs.convert_all(bulk_atoms))
        batch = batch.to(device if device is not None else self.device)

        calculated_batch = self.eval_with_oom_logic(batch, self._batched_static_eval)
        # reset the tags, they got lost in conversion to Torch
        slabs = batch_to_atoms(calculated_batch)
        # collect the reference and adslab energies
        adslab_e = np.array([ats.get_potential_energy() for ats in adslabs])
        slab_ref = np.array([s.get_potential_energy() for s in slabs])
        ads_ref = np.array(ads_e)
        # calculate adsorption energy!

        # adsorption_energy = adslab_e - slab_ref - ads_ref

        # json_fnames_ids = dict()
        # for i, at_name in enumerate(atoms_names):
        #     json_fname = str((self.traj_dir / at_name).parent / "adsorption.json")
        #     json_id = (self.traj_dir / at_name).stem.split("-")[0]

        #     if json_fname in json_fnames_ids.keys():
        #         json_fnames_ids[json_fname].update(
        #             {
        #                 json_id: {
        #                     "adsorption_energy": adslab_e[i],
        #                     "adslab_energy": adslab_e[i],
        #                     "ads_reference_energy": ads_ref[i],
        #                     "slab_reference_energy": slab_ref[i],
        #                 }
        #             }
        #         )
        #     else:
        #         json_fnames_ids[json_fname] = {
        #             json_id: {
        #                 "adsorption_energy": adslab_e[i],
        #                 "adslab_energy": adslab_e[i],
        #                 "ads_reference_energy": ads_ref[i],
        #                 "slab_reference_energy": slab_ref[i],
        #             }
        #         }

        json_fnames = [
            (self.traj_dir / at_name).parent / "adsorption.json"
            for at_name in atoms_names
        ]
        json_ids = [
            (self.traj_dir / at_name).stem.split("-")[0] for at_name in atoms_names
        ]

        # save the results into files to avoid recalculations
        for i, json_fname in enumerate(json_fnames):
            validity = self.adsorbed_structure_checker(adslabs[i])
            self.write_json(
                json_fname,
                {
                    json_ids[i]: {
                        "adsorption_energy": adslab_e[i],
                        "adslab_energy": adslab_e[i],
                        "ads_reference_energy": ads_ref[i],
                        "slab_reference_energy": slab_ref[i],
                        "validity": validity,
                    }
                },
            )

        return adslab_e

    def _batched_static_eval(self, batch):
        """Run static energy/force calculation on batch."""
        self.gnn_calls += 1
        calc = self.get_torch_model
        start = time.time()
        predictions = calc.predict(batch, per_image=False, disable_tqdm=True)
        end = time.time()
        self.gnn_time += end - start
        energy = predictions["energy"]
        forces = predictions["forces"]
        batch.y = energy
        batch.force = forces
        return batch

    @staticmethod
    def eval_with_oom_logic(batch: Batch, method: callable, **kwargs):
        """Evaluate a function on batches with oom error handling.

        OOM error handling from:
        https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/common/relaxation/ml_relaxation.py # noqa
        (8/18/2023)

        The only modification here is the ability to run an aribtrary function.
        """
        batches = deque([batch])
        evaluated_batches = []
        while batches:
            batch = batches.popleft()
            oom = False

            e: Optional[RuntimeError] = None
            try:
                result_batch = method(batch, **kwargs)
                evaluated_batches.append(result_batch)
            except RuntimeError as err:
                e = err
                oom = True
                torch.cuda.empty_cache()

            if oom:
                # move OOM recovery code outside of except clause to allow tensors to
                # be freed.
                data_list = batch.to_data_list()
                if len(data_list) == 1:
                    raise assert_is_instance(e, RuntimeError)
                print(
                    f"Failed to relax batch with size: {len(data_list)}, splitting into two..."  # noqa
                )
                mid = len(data_list) // 2
                batches.appendleft(data_list_collater(data_list[:mid]))
                batches.appendleft(data_list_collater(data_list[mid:]))

        return Batch.from_data_list(evaluated_batches)

    @staticmethod
    def prepare_atoms(atoms: Atoms, constraints: bool = True) -> None:
        """Prepare an atoms object for simulation."""
        if constraints:
            cons = FixAtoms(indices=[atom.index for atom in atoms if (atom.tag == 0)])
            atoms.set_constraint(cons)
        atoms.center(vacuum=13.0, axis=2)
        atoms.set_pbc(True)

    @staticmethod
    def prepare_atoms_list(atoms_list: Atoms, constraints: bool = True) -> None:
        """Prepare an atoms object for simulation."""
        for ats in atoms_list:
            OCAdsorptionCalculator.prepare_atoms(ats, constraints=constraints)

    @staticmethod
    def copy_atoms_list(atoms_list: list[Atoms]) -> list[Atoms]:
        """Copy the atoms in a list and return the copy."""
        return [ats.copy() for ats in atoms_list]

    @staticmethod
    def write_json(fname: Path, data_dict: dict):
        """Write given data dict to json file with exclusive access."""
        written = False
        while not written:
            try:
                with open(str(fname) + "-lock", "x") as f:
                    try:
                        if fname.exists():
                            with open(fname, "r") as f:
                                file_data = json.load(f)
                        else:
                            file_data = {}

                        data_dict.update(
                            file_data
                        )  # Update with runs that have finished
                        with open(fname, "w") as f:
                            json.dump(data_dict, f)

                    except BaseException as err:
                        Path(str(fname) + "-lock").unlink()
                        raise err
                Path(str(fname) + "-lock").unlink()
                written = True
            except FileExistsError:
                pass

    def prediction_path(self, adslab_name):
        """Return the adsorption path for the given adslab."""
        adslab_dir = self.traj_dir / adslab_name
        adslab_dir.mkdir(parents=True, exist_ok=True)
        return adslab_dir / "adsorption.json"

    def get_prediction(self, adslab_name, idx) -> Optional[float]:
        """Get the adsorption energy from adslab_name for given idx.

        If the calculation has not been done, returns None."""
        if self.adsorption_path(adslab_name).exists():
            with open(
                self.adsorption_path(adslab_name),
                "r",
            ) as f:
                data = json.load(f)
            if idx in data.keys() and "adsorption_energy" in data[idx].keys():
                ads_energy = data[idx]["adsorption_energy"]
                return ads_energy
            else:
                return None
        else:
            return None

    def get_validity(self, adslab_name, idx) -> Optional[float]:
        """Get the adsorption energy from adslab_name for given idx.

        If the calculation has not been done, returns None."""
        if self.adsorption_path(adslab_name).exists():
            with open(
                self.adsorption_path(adslab_name),
                "r",
            ) as f:
                data = json.load(f)
            if idx in data.keys() and "validity" in data[idx].keys():
                validity = data[idx]["validity"]
                return validity
            else:
                return None
        else:
            return None

    def slab_path(self, slab_name: str) -> Path:
        """Return the path to the slab file for slab_name."""
        slab_dir = self.traj_dir / "slabs"
        slab_dir.mkdir(parents=True, exist_ok=True)
        return slab_dir / (slab_name + ".pkl")

    def slab_samples_path(self, slab_name: str) -> Path:
        """Return the path to the slab samples file for slab_name."""
        slab_dir = self.traj_dir / "slabs"
        slab_dir.mkdir(parents=True, exist_ok=True)
        return slab_dir / (slab_name + "_samples.pkl")

    def get_slab(self, slab_name: str) -> Optional[float]:
        """Get the slab configuration for the given slab_name.

        If the calculation has not been done, returns None."""
        if self.slab_path(slab_name).exists():
            with open(self.slab_path(slab_name), "rb") as f:
                return pickle.load(f)
        else:
            return None

    def choose_slab(self, slab_samples: list[Atoms], slab_name=None) -> Atoms:
        """Choose the minimum slab from a given set of slabs."""
        atoms = self.copy_atoms_list(slab_samples)
        self.prepare_atoms_list(atoms)
        batch = Batch.from_data_list(self.ats_to_graphs.convert_all(atoms))
        batch = batch.to(self.device)

        calculated_batch = self.eval_with_oom_logic(batch, self._batched_static_eval)
        calculated_slabs = batch_to_atoms(calculated_batch)
        min_idx = np.argmin([s.get_potential_energy() for s in calculated_slabs])

        calculated_slabs[min_idx].info.update(atoms[min_idx].info)
        if slab_name is not None:
            self.save_slab(
                slab_name, calculated_slabs[min_idx], slab_samples=slab_samples
            )
            return self.get_slab(slab_name=slab_name)
        else:
            return calculated_slabs[min_idx]

    def save_slab(self, slab_name: str, slab: Path, slab_samples=None):
        """Save the given slab."""
        try:
            with open(self.slab_path(slab_name), "xb") as f:
                pickle.dump(slab, f)
            if slab_samples is not None:
                with open(self.slab_samples_path(slab_name), "xb") as f:
                    pickle.dump(slab_samples, f)
        except FileExistsError:
            print("Unable to save slab as a slab already exists.")

    def adsorption_path(self, adslab_name):
        """Retunr the path to the adsorption energy file for given adslab."""
        return self.traj_dir / adslab_name / "adsorption.json"


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
        if not self.check_adsorption(ats):
            return self.adsorbate_dissociation_code
        elif not self.check_adsorption(ats):
            return self.desorption_code
        else:
            return self.all_clear_code

    def check_adsorption(self, ats: Atoms):
        """Mesure whether or not the atoms adsorbed."""
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
        conn_matrix = build_neighbor_list(ats).get_connectivity_matrix(sparse=False)
        return np.all(conn_matrix)


def order_of_magnitude(number):
    """Get order of magnitude of a number."""
    return int(np.log10(number))


def break_trajectory(traj_path: Path, dirname: str = None):
    """Break trajectory into a directory of xyz files."""
    if isinstance(traj_path, str):
        traj_path = Path(traj_path)
    if dirname is None:
        dir_path = traj_path.parent / traj_path.stem
    else:
        dir_path = traj_path.parent / dirname
    dir_path.mkdir(parents=True, exist_ok=True)
    [p.unlink() for p in dir_path.rglob("*.xyz")]

    traj = Trajectory(filename=traj_path)
    mag = order_of_magnitude(len(traj))
    for i, ats in enumerate(traj):
        write(dir_path / f"{str(i).zfill(mag+1)}.xyz", ats)
