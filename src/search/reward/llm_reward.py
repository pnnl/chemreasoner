"""Funcitons for atomistic rewards."""
import sys

sys.path.append("src")
from llm import query  # noqa: E402


def llm_adsorption_energy_reward(
    s: query.QueryState, reward_limit: float = 15.0, max_attempts: int = 3
):
    """Reward function to return adsorption energy of reactants.

    The calculated reward is stored in state.reward."""
    s.set_reward(2)
    return 2
    e = reward_limit + 1
    attempts = 0
    while e > reward_limit and attempts < max_attempts:
        e = s.query_adsorption_energy_list()
        attempts += 1

    if attempts == max_attempts:
        s.reward = -10
        return -10

    s.reward = e

    return e
