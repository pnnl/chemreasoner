"""Test the simulation reward workflow."""
import logging
import shutil
import sys
import unittest
import uuid

from pathlib import Path

import numpy as np

from ase import Atoms
from ase.io import read
from ase.data import chemical_symbols


sys.path.append("src")
from llm import ase_interface  # noqa: E402
from search.state.reasoner_state import ReasonerState  # noqa: E402
from evaluation.break_traj_files import break_trajectory  # noqa: E402
from nnp import oc  # noqa: E402
from search.reward.base_reward import BaseReward  # noqa: E402
from search.reward.simulation_reward import StructureReward  # noqa: E402

traj_dir = Path("data", "output", "adsorption_unittest")
sr = StructureReward(
    llm_function=None,
    num_slab_samples=3,
    num_adslab_samples=3,
    **{"model": "gemnet", "traj_dir": traj_dir, "device": "cpu", "steps": 3},
)


class TestBeamSearch(unittest.TestCase):
    """Calculate the reward for answers based on adsorption simulations."""

    def test___init__(TestCase):
        """Select the class of nnp for the reward function."""
        ...

    def test_run_generation_prompts(TestCase):
        """Run the generation prompts for the given states where the reward is None."""
        ...

    def test_run_slab_sym_prompts(TestCase):
        """Run the generation prompts for the given states where the reward is None.

        Updates the given "slab_syms" list in-place.
        """
        ...

    def test___call__(TestCase):
        """Return the calculated adsorption energy from the predicted catalysts."""
        ...

    def test_create_structures_and_calculate(TestCase):
        """Create the structures from the symbols and calculate adsorption energies."""
        answers = sr.create_structures_and_calculate(
            [["Cu"], ["Pt"], ["Zr"]],
            ["CO", "phenol", "anisole"],
            ["Cu", "Pt", "Zr"],
            adsorbate_height=1.87,
        )
        for p in traj_dir.rglob("*.traj"):
            break_trajectory(p)
        assert answers is not None
        # shutil.rmtree(traj_dir)

    def test_parse_adsorption_energies(TestCase):
        """Parse adsorption energies to get the reward value."""
        ...

    def test_create_batches_and_calculate(TestCase):
        """Split adslabs into batches and run the simulations."""
        ...

    def test_unpack_batch_results(TestCase):
        """Unpack a collection of batch results."""
        ...

    def test_calculate_batch(TestCase):
        """Calculate adsorption energies for a batch of atoms objects."""
        ...

    def test_sample_adslabs(TestCase):
        """Sample possible adsorbate+slab combinations."""
        ...

    def test_reduce_metal_symbols(TestCase):
        """Reduce the symbols of metal symbols to a basic form.

        If there are two metals, the more prominant metal is listed first. If there are
        three, the metals are listed in alphabetical order.
        """
        ...

    def test_reduce_candidate_symbols(TestCase):
        """Reduce the symbols of metal symbols to a basic form.

        If there are two metals, the more prominant metal is listed first. If there are
        three, the metals are listed in alphabetical order.
        """
        ...


logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":
    unittest.main()
    shutil.rmtree(traj_dir)
