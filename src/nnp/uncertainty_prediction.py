"""A wrapper for the uncertainty prediction pipeline."""

import sys

from pathlib import Path

from ase import Atoms
from ase.io import read

from ocpmodels.preprocessing.atoms_to_graphs import AtomsToGraphs

from torch_geometric.data import Batch
from torch_geometric.loader.data_list_loader import DataListLoader

sys.path.append("src")
from uq.uq_inference import GBMRegressor, get_per_sample_embeddings
from nnp.oc import OCAdsorptionCalculator


class UncertaintyCalculator:
    def __init__(
        self,
        gnn_calc: OCAdsorptionCalculator,
        model_path,
        lower_alpha=0.1,
        upper_alpha=0.9,
        n_estimators=100,
    ):
        """Create self, storing the given device."""
        self.model_path = model_path
        self.lower_alpha = lower_alpha
        self.upper_alpha = upper_alpha
        self.n_estimators = n_estimators
        self.gbm_model = GBMRegressor(
            model_path=model_path,
            lower_alpha=lower_alpha,
            upper_alpha=upper_alpha,
            n_estimators=n_estimators,
        )
        self.gbm_model._load()
        self.ats_to_graphs = AtomsToGraphs(
            r_edges=False,
            r_fixed=True,
            r_pbc=True,
        )
        self.gnn_calc = gnn_calc

        self.torch_calc = self.gnn_calc.get_torch_model

    def batched_uncertainty_calculation(
        self, atoms: list[Atoms], atoms_names: list[str] = None, device: str = None
    ):
        """Calculate the uncertainty in batches."""
        # Set up calculation for oc
        data_list = self.ats_to_graphs.convert_all(atoms, disable_tqdm=True)
        for i, d in enumerate(data_list):
            d.pbc = d.pbc[None, :]
            d.sid = atoms_names[i] if atoms_names is not None else None
        # convert to torch geometric batch
        uncertainties = []
        dl = DataListLoader(data_list, batch_size=1, shuffle=False)
        for data_list in dl:
            batch = Batch.from_data_list(data_list)
            batch = batch.to("cpu")

            uncertainties += self.get_uq(batch).tolist()
        return uncertainties

    def get_uq(self, batch: Batch):
        """Get the uncertainty values for the given batch of structures."""

        _ = self.torch_calc.predict(batch, per_image=False)
        batch_embeddings = get_per_sample_embeddings(
            self.torch_calc.model.model_outemb, batch
        )
        batch_uq = self.gbm_model.predict(batch_embeddings)
        return batch_uq


if __name__ == "__main__":

    cpu_calc = OCAdsorptionCalculator(
        **{
            "model": "gemnet-oc-22",
            "traj_dir": Path("."),
            "batch_size": 1000,
            "device": "cpu",
            "ads_tag": 2,
            "fmax": 0.05,
            "steps": 250,
        }
    )
    uq_calc = UncertaintyCalculator(
        cpu_calc, "data/uq_model_weights/GBMRegressor-peratom_energy.pkl", 0.1, 0.9, 100
    )

    traj_dir = Path("test", "gnn_test_structures/")
    example_structures = []
    i = 0
    for p in traj_dir.rglob("*.xyz"):
        # print("here")
        ats = read(str(p))
        example_structures.append(ats)
        i += 1
        if i == 8:
            break
    print(len(example_structures))
    print(uq_calc.batched_uncertainty_calculation(example_structures))
