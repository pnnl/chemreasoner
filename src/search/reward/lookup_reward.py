"""Lookup adsorption energy from table."""
import itertools

from pathlib import Path

import numpy as np
import pandas as pd


def permute_candidate_species(candidate):
    """Permute candidate catalyst species in case exact match is not found."""
    oxide = " oxide" if " oxide" in candidate else ""
    candidate.replace(" oxide", "")
    split = candidate.split("-")
    permutations = list(itertools.permutations(split))
    permuted_strings = ["-".join(p) + oxide for p in permutations]
    return permuted_strings


def get_df_adsorption_energy(df, candidate, adsorbate_symbols):
    """Search the dataframe for adsorbate symbols and clean symbols."""
    rows = df[(df["clean_symbols"] == candidate) & (df["ads_symbols"] == candidate)]
    if not rows.empty:
        return -min(rows["e_tot"])
    else:
        for permutation in permute_candidate_species(candidate):
            rows = df[
                (df["clean_symbols"] == permutation) & (df["ads_symbols"] == candidate)
            ]
            if not rows.empty:
                nads = rows["nads"] if "nads" in rows.columns else np.ones((len(rows)))
                val = -min((rows["e_tot"] - rows["e_reference"].filena(-np.inf)) / nads)
                return val if not np.isnan(val) else None
        # If nothing is found, return None
        return None


_df_20 = pd.read_csv(Path("src", "search", "tree_search", "atoms", "oc_20_final.csv"))
_df_22 = pd.read_csv(Path("src", "search", "tree_search", "atoms", "oc_22_final.csv"))


def oc_db_adsorption_energy_reward(state):
    """Reward function to return adsorption energy of reactants."""
    reward = 0
    for i in range(3):
        candidate = state.get_candidate(i).lower()
        candidate.replace("catalysts", "")

        if "oxide" not in candidate:
            # Search oc22
            energy = get_df_adsorption_energy(_df_22, candidate, state.ads_symbols)
        else:
            # Search oc20
            energy = get_df_adsorption_energy(_df_20, candidate, state.ads_symbols)

        if energy is None:
            # Fall back on asking ChatGPT
            reward += -10
        else:
            reward += energy
    return reward
