"""Test the adsorption energy of molecules and catalysts."""
import sys

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from ase import Atoms
from ase.io import read

sys.path.append("src")
from search.reward import simulation_reward  # noqa: E402
from evaluation.break_traj_files import break_trajectory  # noqa: E402


def sumarize_structure(xyz_file: Path):
    """Summarize the information from the given structure."""
    ats = read(xyz_file)

    energy = ats.get_potential_energy()

    adsorbed = simulation_reward.measure_adsorption(ats, cutoff=0.5)

    adslab_string = xyz_file.parent.parent.stem
    catalyst, adsorbate = adslab_string.split("_")
    adsorption_string = xyz_file.parent.parent.parent.name

    starting_distance = float(adsorption_string.split("_")[-1])

    return (
        adsorbed,
        energy,
        catalyst,
        adsorbate,
        starting_distance,
    )


# df = pd.DataFrame(
#     columns=["adsorbed", "ads_energy", "adsorbate", "catalyst", "starting_distance"]
# )
# for p in Path("data", "output", "adsorption_test_11_8_23").rglob("*.traj"):
#     print(p)
#     xyz_directory = p.parent / p.stem
#     if not xyz_directory.exists():
#         break_trajectory(p)
#     xyz_directory = p.parent / p.stem
#     final_structure = sorted(list(xyz_directory.rglob("*.xyz")), key=str)[-1]
#     try:
#         results = sumarize_structure(final_structure)

#     except ValueError as err:
#         if "could not convert string to float" in err.args[0]:
#             continue
#         else:
#             raise err

#     df = pd.concat(
#         [
#             df,
#             pd.DataFrame(
#                 {
#                     "adsorbed": [results[0]],
#                     "ads_energy": [results[1]],
#                     "adsorbate": [results[2]],
#                     "catalyst": [results[3]],
#                     "starting_distance": [results[4]],
#                 }
#             ),
#         ]
#     )

# df.to_csv(Path("data", "output", "adsorption_test_11_8_23", "results.csv"), index=False)

df = pd.read_csv(Path("data", "output", "adsorption_test_11_8_23", "results.csv"))

df["size"] = df["adsorbed"] * 0

print(df)

df["adslab"] = df["adsorbate"] + "_" + df["catalyst"]

sns.scatterplot(
    data=df,
    x="adsorbed",
    y="ads_energy",
    hue="starting_distance",
    style="adslab",
    size="size",
    sizes=(64, 64),
)
plt.ylabel(r"$E_{ads} (eV)$")
plt.xlabel("Relaxed Distance (Å)")
plt.show()
