"""A wrapper for the uncertainty prediction pipeline."""

import random

from ase import Atoms

from ocpmodels.preprocessing.atoms_to_graphs import AtomsToGraphs

from torch_geometric.data import Batch
from torch_geometric.loader.data_list_loader import DataListLoader


class UncertaintyCalculator:
    def __init__(self, device="cuda", batch_size: int = 40):
        """Create self, storing the given device."""
        self.device = device
        self.ats_to_graphs = AtomsToGraphs(
            r_edges=False,
            r_fixed=True,
            r_pbc=True,
        )

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

            uncertainties += [random.random() for _ in range(len(data_list))]
        return uncertainties
