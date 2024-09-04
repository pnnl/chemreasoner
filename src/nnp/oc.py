"""Scripts for running atomic optimizations with ocp-models.

Must intsall the sub-module ocpmodels included in ext/ocp.
"""

import json
import pickle
import sys
import time
import wget
import yaml

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import Trajectory, read, write
from ase.neighborlist import build_neighbor_list
from ase.optimize import BFGS

from ocpmodels.common.relaxation.ase_utils import OCPCalculator, batch_to_atoms
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.typing import assert_is_instance
from ocpmodels.datasets.lmdb_dataset import data_list_collater
from ocpmodels.preprocessing.atoms_to_graphs import AtomsToGraphs


import torch
import torch_geometric
from torch_geometric.data import Batch
from torch_geometric.loader.data_list_loader import DataListLoader

import redis

sys.path.append("src")
from nnp.base_nnp import BaseAdsorptionCalculator


class OCAdsorptionCalculator(BaseAdsorptionCalculator):
    """Class to calculate adsorption energies. Follows Open Catalyst Porject methods."""

    model_weights_paths = Path("data", "model_weights")
    model_weights_paths.mkdir(parents=True, exist_ok=True)
    model_configs_paths = Path("ext", "ocp", "configs")

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
        batch_size=64,
        device="cpu",
        ads_tag=2,
        fmax=0.005,
        steps=150,
        adsorbed_structure_checker=None,
        num_gpus=1,
    ):
        """Create object from model class (gemnet or equiformer).

        Downloads weights if they are not available.
        """
        self.gnn_calls = 0
        self.gnn_time = 0
        self.gnn_relaxed = 0
        self.energies_retrieved = 0
        self.device = device
        # self.device = "cpu"
        self.batch_size = batch_size
        self.fmax = fmax
        self.steps = steps
        self.model = model
        # self.model_weights_paths  = Path("/Users/pana982/models/chemreasoner")
        self.ads_tag = ads_tag  # gihan
        if self.model == "gemnet-t":
            self.model_path = self.model_weights_paths / "gemnet_t_direct_h512_all.pt"
            # print('model path', self.model_path)
            if not self.model_path.exists():
                print("Downloading weights for gemnet...")
                wget.download(
                    "https://dl.fbaipublicfiles.com/opencatalystproject/models/"
                    "2021_08/s2ef/gemnet_t_direct_h512_all.pt",
                    out=str(self.model_weights_paths),
                )
                print("Done!")
            self.config_path = (
                self.model_configs_paths / "s2ef" / "all" / "gemnet" / "gemnet-dT.yml"
            )

        elif self.model == "gemnet-oc-large":
            self.model_path = (
                self.model_weights_paths / "gemnet_oc_large_s2ef_all_md.pt"
            )
            # print('model path', self.model_path)
            if not self.model_path.exists():
                print("Downloading weights for gemnet...")
                wget.download(
                    "https://dl.fbaipublicfiles.com/opencatalystproject/models/"
                    "2022_07/s2ef/gemnet_oc_large_s2ef_all_md.pt",
                    out=str(self.model_weights_paths),
                )
                print("Done!")
            self.config_path = (
                self.model_configs_paths
                / "s2ef"
                / "all"
                / "gemnet"
                / "gemnet-oc-large.yml"
            )

        elif self.model == "gemnet-oc-22":
            self.model_path = self.model_weights_paths / "gnoc_oc22_oc20_all_s2ef.pt"
            # print('model path', self.model_path)
            if not self.model_path.exists():
                print("Downloading weights for gemnet...")
                wget.download(
                    "https://dl.fbaipublicfiles.com/opencatalystproject/models/"
                    "2023_05/oc22/s2ef/gnoc_oc22_oc20_all_s2ef.pt",
                    out=str(self.model_weights_paths),
                )
                print("Done!")
            self.config_path = (
                self.model_configs_paths
                / "oc22"
                / "s2ef"
                / "gemnet-oc"
                / "gemnet_oc_oc20_oc22_degen_edges.yml"
            )

        elif self.model == "escn":
            self.model_path = (
                self.model_weights_paths / "escn_l6_m3_lay20_all_md_s2ef.pt"
            )
            # print('model path', self.model_path)
            if not self.model_path.exists():
                print("Downloading weights for gemnet...")
                wget.download(
                    "https://dl.fbaipublicfiles.com/opencatalystproject/models/"
                    "2023_03/s2ef/escn_l6_m3_lay20_all_md_s2ef.pt",
                    out=str(self.model_weights_paths),
                )
                print("Done!")
            self.config_path = (
                self.model_configs_paths
                / "s2ef"
                / "all"
                / "escn"
                / "eSCN-L6-M3-Lay20-All-MD.yml"
            )

        elif self.model == "eq2":
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
                / "s2ef"
                / "all"
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

        self.redis_db = redis.Redis(host="localhost", port=6379, db=1)

    @property
    def get_ase_calculator(self):
        """Return an ase calculator for self.

        Specifying device overrides self.device.
        """
        if self.ase_calc is None:
            # set up calculator for ase relaxations
            self.ase_calc = OCPCalculator(
                config_yml=str(self.config_path),
                checkpoint=str(self.model_path),
                trainer="forces",
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
            if self.device != "cpu":
                self.torch_calc.model = BatchDataParallelPassthrough(
                    self.torch_calc.model
                )
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
        atoms_names: list[str],
        device: str = None,
        fmax: float = None,
        steps: int = None,
        constraints: bool = True,
        **bfgs_kwargs,
    ):
        """Relax the postitions of the given atoms. Setting device overrides self."""
        atoms = self.copy_atoms_list(atoms)
        fmax = fmax if fmax is not None else self.fmax
        steps = steps if steps is not None else self.steps
        # Set up calculation for oc
        self.prepare_atoms_list(
            atoms, constraints=constraints
        )  # list[ase] -> list[Data]
        data_list = self.ats_to_graphs.convert_all(atoms, disable_tqdm=True)
        for i, d in enumerate(data_list):
            d.pbc = d.pbc[None, :]
            d.sid = atoms_names[i]
        # convert to torch geometric batch
        final_atoms = []
        s_ids = []
        dl = DataListLoader(data_list, batch_size=self.batch_size, shuffle=False)
        for data_list in dl:
            batch = Batch.from_data_list(data_list)
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
                # device="cuda:1",
            )
            end = time.time()
            self.gnn_calls += self.steps
            self.gnn_relaxed += len(atoms)
            self.gnn_time += end - start

            s_ids += batch.sid
            final_atoms += batch_to_atoms(final_batch)

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
                if t == self.ads_tag:  # part of the adsorbate
                    e_ref += self.ads_references[ats.get_atomic_numbers()[i]]
                else:  # part of the bulk
                    bulk_ats.append(ats[i])
            ads_e.append(e_ref)
            bulk_atoms.append(bulk_ats.copy())
        # convert to torch geometric batch
        data_list = self.ats_to_graphs.convert_all(bulk_atoms, disable_tqdm=True)
        for i, d in enumerate(data_list):
            d.pbc = d.pbc[None, :]

        batch = Batch.from_data_list(data_list)

        # device='cpu'
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

    def static_eval(self, atoms: list[Atoms], device: str = None):
        """Evaluate the static energies and forces of the given atoms."""
        data_list = self.ats_to_graphs.convert_all(atoms, disable_tqdm=True)
        for i, d in enumerate(data_list):
            d.pbc = d.pbc[None, :]
        batch = Batch.from_data_list(data_list)

        batch = batch.to(device if device is not None else self.device)
        calculated_batch = self.eval_with_oom_logic(batch, self._batched_static_eval)
        # reset the tags, they got lost in conversion to Torch
        return batch_to_atoms(calculated_batch)

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

    def hessian_jacobian_f(self, atoms: Atoms, constraints=True, device: str = None):
        """Compute the hessian matrix for the given set of structures by taking jacobian of F."""
        atoms = self.copy_atoms_list(atoms)
        self.prepare_atoms_list(
            atoms, constraints=constraints
        )  # list[ase] -> list[Data]
        data_list = self.ats_to_graphs.convert_all(atoms, disable_tqdm=True)
        for i, d in enumerate(data_list):
            d.pbc = d.pbc[None, :]
            # d.sid = i
        # convert to torch geometric batch
        dl = DataListLoader(data_list, batch_size=self.batch_size, shuffle=False)
        hessians = []
        for data_list in dl:
            batch = Batch.from_data_list(data_list)
            batch = batch.to(device if device is not None else self.device)

            def _batch_wrapper(pos: torch.Tensor):
                # batch.pos = torch.autograd.Variable(pos, requires_grad=True)
                batch.pos.requires_grad_(True)
                batch.pos = pos
                f = -self.get_torch_model.model.forward(
                    batch,
                )["forces"]
                print(f.requires_grad)
                print(f)
                print(f.shape)
                print(type(f))
                return f

            def _hessian_wrapper(b: Batch):
                H = torch.autograd.functional.jacobian(
                    _batch_wrapper,
                    inputs=b.pos,
                )
                print(H.shape)
                b.hessian = H
                return b

            hessians.append(
                self.eval_with_oom_logic(
                    batch,
                    _hessian_wrapper,
                )
            )
        return hessians

    def hessian_grad_sqr_e(self, atoms: Atoms, constraints=True, device: str = None):
        """Compute the hessian matrix for the given set of structures by taking Hess of E."""
        atoms = self.copy_atoms_list(atoms)
        self.prepare_atoms_list(
            atoms, constraints=constraints
        )  # list[ase] -> list[Data]
        data_list = self.ats_to_graphs.convert_all(atoms, disable_tqdm=True)
        for i, d in enumerate(data_list):
            d.pbc = d.pbc[None, :]
            # d.sid = i
        # convert to torch geometric batch
        dl = DataListLoader(data_list, batch_size=self.batch_size, shuffle=False)
        hessians = []
        for data_list in dl:
            batch = Batch.from_data_list(data_list)
            batch = batch.to(device if device is not None else self.device)

            def _batch_wrapper(pos: torch.Tensor):
                # batch.pos = torch.autograd.Variable(pos, requires_grad=True)
                batch.pos.requires_grad_(True)
                batch.pos = pos
                e = -self.get_torch_model.model.forward(
                    batch,
                )["energy"]
                print(e.requires_grad)
                print(e)
                print(e.shape)
                print(type(e))
                return e

            def _hessian_wrapper(b: Batch):
                H = torch.autograd.functional.hessian(
                    _batch_wrapper,
                    inputs=b.pos,
                )
                print(H.shape)
                b.hessian = H
                return b

            hessians.append(
                self.eval_with_oom_logic(
                    batch,
                    _hessian_wrapper,
                )
            )
        return hessians

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
                    f"Failed to calculate batch with size: {len(data_list)}, splitting into two..."  # noqa
                )
                mid = len(data_list) // 2
                batches.appendleft(data_list_collater(data_list[:mid]))
                batches.appendleft(data_list_collater(data_list[mid:]))

        return Batch.from_data_list(evaluated_batches)

    @staticmethod
    def prepare_atoms(atoms: Atoms, constraints: bool = True) -> None:
        """Prepare an atoms object for simulation."""
        if constraints and 8 not in [atom.number for atom in atoms if (atom.tag != 2)]:
            cons = FixAtoms(indices=[atom.index for atom in atoms if (atom.tag == 0)])
            atoms.set_constraint(cons)
        atoms.center(vacuum=13.0, axis=2)
        atoms.set_pbc(True)

    @staticmethod
    def prepare_atoms_list(atoms_list: Atoms, constraints: bool = True) -> None:
        """Prepare an atoms object list for simulation."""
        for ats in atoms_list:
            OCAdsorptionCalculator.prepare_atoms(ats, constraints=constraints)

    @staticmethod
    def copy_atoms_list(atoms_list: list[Atoms]) -> list[Atoms]:
        """Copy the atoms in a list and return the copy."""
        return [ats.copy() for ats in atoms_list]

    # @staticmethod
    # def write_json_deprecated(fname: Path, data_dict: dict):
    #     """Write given data dict to json file with exclusive access."""
    #     written = False
    #     while not written:
    #         try:
    #             with open(str(fname) + "-lock", "x") as f:
    #                 try:
    #                     if fname.exists():
    #                         with open(fname, "r") as f:
    #                             file_data = json.load(f)
    #                     else:
    #                         file_data = {}

    #                     data_dict.update(
    #                         file_data
    #                     )  # Update with runs that have finished
    #                     with open(fname, "w") as f:
    #                         json.dump(data_dict, f)

    #                 except BaseException as err:
    #                     Path(str(fname) + "-lock").unlink()
    #                     raise err
    #             Path(str(fname) + "-lock").unlink()
    #             written = True
    #         except FileExistsError:
    #             pass

    def write_json(self, fname: Path, data_dict: dict):
        """Write given data dict to json file with exclusive access."""
        data = self.read_json(fname)

        if data is None:
            self.redis_db.set(str(fname), json.dumps(data_dict))
        else:
            data.update(data_dict)
            self.redis_db.set(str(fname), json.dumps(data))

        with open(fname, "w") as f:
            json.dump(data, f)

    def read_json(self, fname: Path):
        """Write given data dict to json file with exclusive access."""
        data = self.redis_db.get(str(fname))
        if data is not None:
            return json.loads(self.redis_db.get(str(fname)))
        else:
            return None

    def prediction_path(self, adslab_name):
        """Return the adsorption path for the given adslab."""
        adslab_dir = self.traj_dir / adslab_name
        adslab_dir.mkdir(parents=True, exist_ok=True)
        return adslab_dir / "adsorption.json"

    def get_prediction_deprecated(self, adslab_name, idx) -> Optional[float]:
        """Get the adsorption energy from adslab_name for given idx.

        If the calculation has not been done, returns None."""
        if self.adsorption_path(adslab_name).exists():
            data = self.read_json(self.adsorption_path(adslab_name))
            if idx in data.keys() and "adsorption_energy" in data[idx].keys():
                ads_energy = data[idx]["adsorption_energy"]
                return ads_energy
            else:
                return None
        else:
            return None

    def get_prediction(self, adslab_name, idx) -> Optional[float]:
        """Get the adsorption energy from adslab_name for given idx.

        If the calculation has not been done, returns None."""

        data = self.read_json(self.adsorption_path(adslab_name))
        print(self.adsorption_path(adslab_name))
        print(data)
        if (
            data is not None
            and idx in data.keys()
            and "adsorption_energy" in data[idx].keys()
        ):
            ads_energy = data[idx]["adsorption_energy"]
            self.energies_retrieved += 1
            return ads_energy
        else:
            return None

    def get_validity_deprecated(self, adslab_name, idx) -> Optional[float]:
        """Get the adsorption energy from adslab_name for given idx.

        If the calculation has not been done, returns None."""
        if self.adsorption_path(adslab_name).exists():
            data = self.read_json(self.adsorption_path(adslab_name))
            if idx in data.keys() and "validity" in data[idx].keys():
                validity = data[idx]["validity"]
                return validity
            else:
                return None
        else:
            return None

    def get_validity(self, adslab_name, idx) -> Optional[float]:
        """Get the adsorption energy from adslab_name for given idx.

        If the calculation has not been done, returns None."""
        data = self.read_json(self.adsorption_path(adslab_name))
        print(self.adsorption_path(adslab_name))
        print(data)
        if data is not None and idx in data.keys() and "validity" in data[idx].keys():
            validity = data[idx]["validity"]
            return validity
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
        data = self.redis_db.get(str(self.slab_path(slab_name)))
        if data is not None:
            return pickle.loads(data)
        else:
            return None

        if self.slab_path(slab_name).exists():
            with open(self.slab_path(slab_name), "rb") as f:
                return pickle.load(f)
        else:
            return None

    def choose_slab(self, slab_samples: list[Atoms], slab_name=None) -> Atoms:
        """Choose the minimum slab from a given set of slabs."""
        atoms = self.copy_atoms_list(slab_samples)
        self.prepare_atoms_list(atoms)
        data_list = self.ats_to_graphs.convert_all(atoms, disable_tqdm=True)
        for i, d in enumerate(data_list):
            d.pbc = d.pbc[None, :]

        batch = Batch.from_data_list(data_list)
        batch = batch.to(self.device)
        print(batch)

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
            self.redis_db.set(str(self.slab_path(slab_name)), pickle.dumps(slab))

            if slab_samples is not None:
                with open(self.slab_samples_path(slab_name), "xb") as f:
                    pickle.dump(slab_samples, f)
        except FileExistsError:
            print("Unable to save slab as a slab already exists.")

    def adsorption_path(self, adslab_name):
        """Retunr the path to the adsorption energy file for given adslab."""
        return self.traj_dir / adslab_name / "adsorption.json"

    def calculate_slab_energies(self, structures: list[Atoms], device: str = None):
        """Calculate the slab energies for the given structures."""
        # Set up calculation for oc
        bulk_atoms = []
        for ats in structures:
            bulk_ats = Atoms()
            bulk_ats.set_cell(ats.get_cell())
            bulk_ats.set_pbc(ats.get_pbc())
            for i, t in enumerate(ats.get_tags()):
                if t != self.ads_tag:  # part of the adsorbate
                    bulk_ats.append(ats[i])

            bulk_atoms.append(bulk_ats.copy())

        data_list = self.ats_to_graphs.convert_all(bulk_atoms, disable_tqdm=True)
        for i, d in enumerate(data_list):
            d.pbc = d.pbc[None, :]
        # convert to torch geometric batch
        batch = Batch.from_data_list(data_list)

        # device='cpu'
        batch = batch.to(device if device is not None else self.device)

        calculated_batch = self.eval_with_oom_logic(batch, self._batched_static_eval)
        # reset the tags, they got lost in conversion to Torch
        slabs = batch_to_atoms(calculated_batch)
        # collect the reference and adslab energies
        return slabs

    def calculate_adsorption_energies(
        self,
        trajectories: list[Atoms],
        device: str = None,
        compute_final_energy: bool = True,
    ):
        """The calculate the adsorption energies for each trajectory.

        Slab reference calculations are performed using the first entry in trajectories.
        Total energy calculations will be performed using the final entry in
        trajectories, as long as compute_final_energy is True.
        """
        if compute_final_energy:
            total_energy_ats = self.static_eval(
                [t[-1] for t in trajectories], device=device
            )

        slab_reference_energy_ats = self.calculate_slab_energies(
            [t[0] for t in trajectories], device=device
        )

        return [
            t.get_potetial_energy() - s.get_potential_energy()
            for t, s in zip(total_energy_ats, slab_reference_energy_ats)
        ]


class AdsorbedStructureChecker:
    """A class to check whether an adsorbed structure is correct or not."

    Uses convention created by Open Catalysis:
    https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md

    0 - no anomaly
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
        return 0
        if not self.measure_dissociation(ats):
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


class BatchDataParallel(torch_geometric.nn.data_parallel.DataParallel):
    """Torch geometric DataParallel class for models with batch inputs."""

    def forward(self, batch: Batch):
        """Convert batch to datalist and perform the usual forward call."""
        if isinstance(batch, Batch):
            data_list = batch.to_data_list()
        else:
            data_list = batch
        if self.device_ids and len(data_list) < len(self.device_ids):
            # if len(batch) <  len(gpus), run batch on first gpu
            data = Batch.from_data_list(
                data_list,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            ).to(self.src_device)
            return self.module(data)
        return super().forward(data_list)


class BatchDataParallelPassthrough(BatchDataParallel):
    """A class to allow the passthrough of custom methods for nn modules in DataParallel.

    Suggested by github user dniku:
    https://github.com/pytorch/pytorch/issues/16885#issuecomment-551779897
    """

    def __getattr__(self, name):
        """Try using DataParallel methods, which to stored module if failed."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


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


if __name__ == "__main__":
    # example_structure = read(
    #     str(
    #         Path(
    #             "test",
    #             "gnn_test_structures",
    #             "0b7c2e76-aa78-4c17-911d-cd6c9fc67d2a.xyz",
    #         )
    #     )
    # )
    # example_structures = [example_structure]

    # calc = OCAdsorptionCalculator(
    #     **{
    #         "model": "gemnet-oc-22",
    #         "traj_dir": Path("data_parallel_benchmark"),
    #         "batch_size": 64,
    #         "device": "cuda",
    #         "ads_tag": 2,
    #         "fmax": 0.05,
    #         "steps": 250,
    #     }
    # )
    gpu_calc = OCAdsorptionCalculator(
        **{
            "model": "gemnet-oc-22",
            "traj_dir": Path(
                "cu_zn_dft_structures", "trajectories_continued_convergence"
            ),
            "batch_size": 1,
            "device": "cuda",
            "ads_tag": 2,
            "fmax": 0.03,
            "steps": 250,
        }
    )
    with open("convergence_error_codes.json", "r") as f:
        codes = json.load(f)
    structures = []
    names = []
    for p in Path("cu_zn_dft_structures", "trajectories_e_tot").rglob("*.xyz"):
        print(p)
        if codes[p.stem] == 6:
            print(p)
            structures.append(read(str(p)))
            names.append(p.stem)
    lengths = {}
    for p in Path("cu_zn_dft_structures", "trajectories_continued_convergence").rglob(
        "*.traj"
    ):
        print(p)
        traj = Trajectory(str(p))
        lengths[p.stem] = len(traj)
    # gpu_calc.batched_relax_atoms(structures, names)
    with open(
        Path("cu_zn_dft_structures", "trajectories_continued_convergence")
        / "lengths.json",
        "w",
    ) as f:
        json.dump(lengths, f)
    # cpu_calc = OCAdsorptionCalculator(
    #     **{
    #         "model": "gemnet-oc-22",
    #         "traj_dir": Path("data_parallel_benchmark"),
    #         "batch_size": 1,
    #         "device": "cpu",
    #         "ads_tag": 2,
    #         "fmax": 0.05,
    #         "steps": 250,
    #     }
    # )
    # hessians_dir = Path("cu_zn_check_relaxation", "hessians").mkdir(
    #     parents=True, exist_ok=True
    # )
    # traj_dir = Path("cu_zn_check_relaxation", "trajectories")
    # for p in traj_dir.rglob("*.traj"):
    #     traj = Trajectory(str(p))
    #     example_structures = [traj[-1]]

    #     try:
    #         start = time.time()
    #         b = gpu_calc.hessian_jacobian_f(
    #             example_structures,
    #         )[0]

    #         end = time.time()
    #         torch.save(
    #             b.hessian, str(p.parent.parent.parent / "hessians" / (p.stem + ".pt"))
    #         )
    #         torch.save(
    #             torch.Tensor([end - start]),
    #             p.parent.parent.parent / "hessians" / (p.stem + "cuda_time.pt"),
    #         )
    #         print(end - start)
    #     except Exception:
    #         torch.cuda.empty_cache()
    #         start = time.time()
    #         b = cpu_calc.hessian_jacobian_f(
    #             example_structures,
    #         )[0]

    #         end = time.time()
    #         torch.save(
    #             b.hessian, str(p.parent.parent.parent / "hessians" / (p.stem + ".pt"))
    #         )
    #         torch.save(
    #             torch.Tensor([end - start]),
    #             str(p.parent.parent.parent / "hessians" / (p.stem + "cpu_time.pt")),
    #         )
    #         print(end - start)

    # start = time.time()
    # b = calc.hessian_grad_sqr_e(
    #     example_structures,
    # )[0]

    # print(b.hessian.shape)

    # # print(b.hessian.reshape(-1, 198)[b.hessian.reshape(-1, 198) != 0])
    # end = time.time()
    # torch.save(b.hessian.detach(), "hessian_e.pt")
    # torch.save(torch.Tensor([end - start]), "hessian_e_time.pt")
    # print((calc.get_torch_model.model))
    # print(torch.nn.DataParallel(calc.get_torch_model))
    # print(torch.nn.DataParallel(calc.get_torch_model.model))

    # print(dir(calc.get_torch_model))
    # print(dir(calc.get_torch_model.model))
