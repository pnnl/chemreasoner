"""A base class for nnp methods."""
from abc import ABCMeta, abstractmethod

from pathlib import Path


class BaseAdsorptionCalculator(metaclass=ABCMeta):
    """A base class for reward functions."""

    traj_dir: Path

    @abstractmethod
    def batched_relax_atoms(self, batch, atoms_names):
        """Relax a batch of structures."""
        ...

    @abstractmethod
    def batched_adsorption_calculation(self, batch, atoms_names):
        """Claculate the adsorption eenrgy from relaxed atomic positions."""
        ...
