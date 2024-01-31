"""Run the sweep over gnn calculated structures."""
import json
import sys

from pathlib import Path
from tqdm import tqdm

from ase.io import Trajectory

sys.path.append("src")
from nnp.oc import OCAdsorptionCalculator  # noqa:E402

ads_calc = OCAdsorptionCalculator(
    **{
        "model": "gemnet-t",
        "traj_dir": Path("gnn_eval_structures", "new_trajs"),
        "batch_size": 40,
        "device": "cuda",
        "ads_tag": 2,
        "fmax": 0.05,
        "steps": 300,
    }
)

atoms = []
atoms_names = []
for p in tqdm(Path("gnn_eval_structures").rglob("*.traj")):
    traj = Trajectory(str(p))
    atoms.append(traj[0])
    atoms_names.append(p.stem)

ads_calc.batched_relax_atoms(atoms, atoms_names)

final_data = {}

for p in tqdm(Path("gnn_eval_structures", "new_trajs").rglob("*.traj")):
    traj = Trajectory(str(p))
    final_data[str(p)] = [ats.get_potential_energy() for ats in traj]

with open(Path("gnn_eval_structures", "new_trajs") / "results.json", "w") as f:
    json.dump(final_data, f)
