"""Pull out the best structures."""
import json
import shutil

from pathlib import Path

gnn_calcs_path = "/dev/shm/chemreasoner/catalysis/"

catalysts = ["CuZnAl", "CuNiZn", "CuFeZn", "NiZnAl", "FeCoZn", "CuZn", "Ru", "Rh", "Pd"]
adsorbates = ["CO2", "*OCHO", "*CHOH", "*OHCH3"]

for cat in catalysts:
    if len(cat) == 6 and cat[2] >= cat[4]:
        cat = cat[0:2] + cat[4:] + cat[2:4]

    for ads in adsorbates:
        adslab_string = f"{cat}_{ads}"
        adslab_path = Path(gnn_calcs_path, adslab_string)
        # print(adslab_path.exists())
        json_path = adslab_path / "adsorption.json"
        with open(json_path, "r") as f:
            data = json.load(f)

        min_idx = min(list(data.keys()), key=lambda k: data[k]["adsorption_energy"])

        traj_path = list(adslab_path.rglob(f"{min_idx}-*.traj"))[0]

        # print(adslab_string)
        (
            Path("/people/spru445/methanol_results")
            / traj_path.parent.stem
            / traj_path.stem
        ).mkdir(parents=True, exist_ok=True)
        shutil.copy(
            traj_path,
            Path("/people/spru445/methanol_results")
            / traj_path.parent.stem
            / (min_idx + ".traj"),
        )
