"""Script to test GemNet on the OC dataset."""

import argparse
import logging
import sys

from pathlib import Path

from ase import Atoms
from ase.io import read

sys.path.append("src")
from nnp.oc import OCAdsorptionCalculator

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument("--cpu", action="store_true")
parser.add_argument("--data_path", type=str)
parser.add_argument("--steps", type=int)
args = parser.parse_args()


data_path = Path(args.data_path)

calc = OCAdsorptionCalculator(
    **{
        "model": "gemnet-oc-22",
        "traj_dir": data_path,
        "batch_size": 32,
        "device": "cpu" if args.cpu else "cuda",
        "ads_tag": 2,
        "fmax": 0.03,
        "steps": args.steps,
    }
)
atoms = {}
for p in data_path.rglob("*.xyz"):
    atoms[p.stem] = read(str(p))

# print(8 not in [atom.number for atom in atoms['oc_20/0_23101'] if (atom.tag == 0)])

timing = {}
energies = {}
keys = list(atoms.keys())

(data_path / "trajectories_e_tot").mkdir(parents=True, exist_ok=True)
atoms_list = []
atoms_names = []
for k, v in atoms.items():
    atoms_list.append(v)
    name = Path("trajectories_e_tot") / k
    atoms_names.append(str(name))
    name.parent.mkdir(parents=True, exist_ok=True)

start_timing = calc.gnn_time
relaxed_atoms = calc.batched_relax_atoms(atoms_list, atoms_names=atoms_names)
end_timing = calc.gnn_time
logging.info(f"Total Energy: {end_timing - start_timing}")
