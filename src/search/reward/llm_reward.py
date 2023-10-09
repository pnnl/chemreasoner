"""Funcitons for atomistic rewards."""
import sys

sys.path.append("src")
from llm import query  # noqa: E402


def llm_adsorption_energy_reward(
    states: list[query.QueryState],
    reward_limit: float = 10.0,
    max_attempts: int = 3,
    primary_reward: bool = True,
):
    """Reward function to return adsorption energy of reactants.

    The calculated reward is stored in state.reward."""
    rewards = []
    for s in states:
        e = reward_limit + 1
        attempts = 0
        while e > reward_limit and attempts < max_attempts:
            e = s.query_adsorption_energy_list(allow_requery=primary_reward)
            attempts += 1

        if attempts == max_attempts:
            e = -10

        if primary_reward:
            s.set_reward(e, info_field="llm-reward")
        else:
            s.set_reward(e, primary_reward=primary_reward, info_field="llm-reward")

        rewards.append(e)

    return rewards
