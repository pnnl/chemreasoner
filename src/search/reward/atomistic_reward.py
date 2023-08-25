"""Modeul for atomistic reward functions. Safe to ignore this."""
from ase import Atoms
from ase.calculators.lj import LennardJones


def lj_reward(state: Atoms, action=None):
    """Reward function to return potential energy."""
    if action is not None:
        state = action(state)
    state = state.copy()
    state.set_calculator(LennardJones())
    return -state.get_potential_energy()
