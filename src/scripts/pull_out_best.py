"""Pull out the best structures."""
import json
import shutil

from pathlib import Path

gnn_calcs_path = "/qfs/projects/aqe_tec4/catalysis/chemreasoner/catalysis/"

catalysts = ["CuZnAl", "CuNiZn", "CuFeZn", "NiZnAl", "FeCoZn"]
adsorbates = ["CO2", "*OCHO", "CHOH", "*OHCH3"]

for cat in catalysts:
    for ads in adsorbates:
        adslab_string = f"{cat}_{ads}"
        adslab_path = Path(gnn_calcs_path, adslab_string)
        print(adslab_path.exists())
        json_path = adslab_path / "adsorption.json"
        with open(json_path, "r") as f:
            data = json.load(f)

        min_idx = min(list(data.keys()), key=lambda k: data[k]["adsorption_energy"])

        traj_path = list(adslab_path.rglob(f"{min_idx}-*.traj"))[0]

        print(adslab_string)
        shutil.copy(
            traj_path, Path("/people/spru445/methanol_results") / traj_path.stem
        )