"""A wrapper for the uncertainty prediction pipeline."""

import random
import sys

from ase import Atoms

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
        save_dir,
        lower_alpha=0.1,
        upper_alpha=0.9,
        n_estimators=100,
    ):
        """Create self, storing the given device."""
        self.savedir = save_dir
        self.lower_alpha = lower_alpha
        self.upper_alpha = upper_alpha
        self.n_estimators = n_estimators
        self.gbm_model = GBMRegressor(
            savedir=save_dir,
            lower_alpha=lower_alpha,
            upper_alpha=upper_alpha,
            n_estimators=n_estimators,
        )
        self.ats_to_graphs = AtomsToGraphs(
            r_edges=False,
            r_fixed=True,
            r_pbc=True,
        )
        self.gnn_calc = gnn_calc

        self.torch_calc = self.gnn_calc.get_torch_model

    def batched_uncertainty_calculation(
        self, atoms: list[Atoms], atoms_names: list[str], device: str = None
    ):
        """Calculate the uncertainty in batches."""
        # Set up calculation for oc
        data_list = self.ats_to_graphs.convert_all(atoms, disable_tqdm=True)
        for i, d in enumerate(data_list):
            d.pbc = d.pbc[None, :]
            d.sid = atoms_names[i]
        # convert to torch geometric batch
        uncertainties = []
        dl = DataListLoader(data_list, batch_size=self.batch_size, shuffle=False)
        for data_list in dl:
            batch = Batch.from_data_list(data_list)
            batch = batch.to(device if device is not None else self.device)

            uncertainties += self.get_uq(batch)
        return uncertainties

    def get_uq(self, batch: Batch):
        """Get the uncertainty values for the given batch of structures."""

        _ = self.torch_calc.predict(batch, per_image=False)
        batch_embeddings = get_per_sample_embeddings(
            self.torch_calc.model.model_outemb, batch
        )
        batch_uq = self.gbm_model.predict(batch_embeddings)
        print(batch_uq)
        return batch_uq
