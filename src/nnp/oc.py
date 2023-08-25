"""Scripts for running atomic optimizations with ocp-models.

Must intsall the sub-module ocpmodels included in ext/ocp.
"""
import json
import wget
import yaml

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import Trajectory, write
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
    model_configs_paths = Path("ext", "ocp", "configs", "s2ef", "all")

    # (8/18/2023) reference values from:
    # https://arxiv.org/abs/2010.09990
    ads_references = {
        1: -3.477,
        5: -8.083,
        6: -7.282,
        8: -7.204,
    }

    def __init__(self, model: str, traj_dir: Path, batch_size=32, device="cpu"):
        """Createobject from model class (gemnet or equiformer).

        Downloads weights if they are not available.
        """
        self.device = device
        self.batch_size = batch_size
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
                / "gemnet"
                / "equiformer_v2_N@20_L@6_M@3_153M.yml"
            )

        else:
            raise ValueError(f"Unkown model {self.model}.")
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.traj_dir = traj_dir
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        # set up atoms to graphs object
        self.ats_to_graphs = AtomsToGraphs(
            r_edges=False,
            r_fixed=True,
            r_pbc=True,
        )

    def get_ase_calculator(self, device=None):
        """Return an ase calculator for self.

        Specifying device overrides self.device.
        """
        # set up calculator for ase relaxations
        ase_calc = OCPCalculator(
            config_yml=str(self.config_path),
            checkpoint=str(self.model_path),
            cpu=self.device == "cpu" if device is None else device == "cpu",
        )
        return ase_calc

    def get_torch_model(self, device=None):
        """Return an ase calculator for self.

        Specifying device overrides self.device.
        """
        # set up calculator for ase relaxations
        ase_calc = OCPCalculator(
            config_yml=str(self.config_path),
            checkpoint=str(self.model_path),
            cpu=self.device == "cpu" if device is None else device == "cpu",
        )
        return ase_calc.trainer

    def relax_atoms_ase(
        self,
        atoms: Atoms,
        device=None,
        fname=None,
        fmax=0.005,
        steps=100,
        **bfgs_kwargs,
    ):
        """Relax the postitions of the given atoms.

        Setting device overrides self.device
        """
        atoms = atoms.copy()
        self.prepare_atoms(atoms)

        atoms.set_calculator(self.get_ase_calculator(device=device))
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
        fmax=0.005,
        steps=100,
        **bfgs_kwargs,
    ):
        """Relax the postitions of the given atoms. Setting device overrides self."""
        atoms = self.copy_atoms_list(atoms)
        # Set up calculation for oc
        for ats in atoms:
            self.prepare_atoms(ats)
        # convert to torch geometric batch
        batch = Batch.from_data_list(self.ats_to_graphs.convert_all(atoms))
        batch.sid = atoms_names
        print(f"Batch sids {batch.sid}")

        trainer = self.get_torch_model(device=device)

        try:
            relax_opt = self.config["task"]["relax_opt"]
        except KeyError:
            relax_opt = {"memory": 100}  # only need to set memory and traj_dir

        relax_opt["traj_dir"] = self.traj_dir
        final_batch = ml_relax(
            batch=[batch],  # ml_relax always uses batch[0]
            model=trainer,
            steps=steps,
            fmax=fmax,
            relax_opt=relax_opt,
            save_full_traj=True,
            device=trainer.device,
        )
        final_atoms = batch_to_atoms(final_batch)
        final_atoms[0].get_potential_energy
        return final_atoms

    def batched_adsorption_calculation(
        self,
        atoms: list[Atoms],
        atoms_names,
        device=None,
        fmax=0.005,
        steps=100,
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

        calculated_batch = self.eval_with_oom_logic(batch, self._batched_static_eval)
        # reset the tags, they got lost in conversion to Torch
        slabs = batch_to_atoms(calculated_batch)
        # collect the reference and adslab energies
        adslab_e = np.array([ats.get_potential_energy() for ats in adslabs])
        slab_ref = np.array([s.get_potential_energy() for s in slabs])
        ads_ref = np.array(ads_e)
        # calculate adsorption energy!

        adsorption_energy = adslab_e - slab_ref - ads_ref

        json_fnames = [
            (self.traj_dir / at_name).parent / "adsorption.json"
            for at_name in atoms_names
        ]
        json_ids = [(self.traj_dir / at_name).stem for at_name in atoms_names]

        # save the results into files to avoid recalculations
        for i, json_fname in enumerate(json_fnames):
            self.update_json(
                json_fname,
                {
                    json_ids[i]: {
                        "adsorption_energy": adsorption_energy[i],
                        "adslab_energy": adslab_e[i],
                        "ads_reference_energy": ads_ref[i],
                        "slab_reference_energy": slab_ref[i],
                    }
                },
            )

        return adsorption_energy

    def _batched_static_eval(self, batch):
        """Run static energy/force calculation on batch."""
        calc = self.get_torch_model()
        predictions = calc.predict(batch, per_image=False, disable_tqdm=True)
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
    def prepare_atoms(adslab: Atoms) -> None:
        """Prepare an atoms object for simulation."""
        cons = FixAtoms(indices=[atom.index for atom in adslab if (atom.tag > 1)])
        adslab.set_constraint(cons)
        adslab.center(vacuum=13.0, axis=2)
        adslab.set_pbc(True)

    @staticmethod
    def copy_atoms_list(atoms_list: list[Atoms]) -> list[Atoms]:
        """Copy the atoms in a list and return the copy."""
        return [ats.copy() for ats in atoms_list]

    @staticmethod
    def update_json(fname, update_dict):
        """Update given json file with update_dict."""
        if fname.exists():
            with open(fname, "r") as f:
                data = json.load(f)

        else:
            data = dict()
        data.update(update_dict)
        with open(fname, "w") as f:
            json.dump(data, f)


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
