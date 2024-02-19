"""Evaluate relaxed adsorption eneries for the given xyz files."""

import sys

from pathlib import Path

from ase.io import read
import pandas as pd

sys.path.append("src")
from nnp.oc import OCAdsorptionCalculator  # noqa:E402

data_path = Path("src/nnp/adslabs")

calc = OCAdsorptionCalculator(
    **{
        "model": "gemnet-t",
        "traj_dir": data_path,
        "batch_size": 32,
        "device": "cuda",
        "ads_tag": 2,
        "fmax": 0.05,
        "steps": 300,
    }
)

batch_size = 40
batch = []
fnames = []
evals = []
for xyz in sorted(data_path.rglob("*.xyz")):
    print(xyz)
    traj_path = str(xyz).replace(str(data_path), "")[1:]
    fnames.append(traj_path.replace(".xyz", ".traj"))
    print(data_path / traj_path)
    ats = read(str(xyz))
    batch.append(ats)
    if len(batch) == batch_size:
        print("=== Running Batch ===")
        calc.batched_relax_atoms(ats, fnames)
        batch = []
        fnames = []

if len(batch) > 0:
    print("=== Running Final Batch ===")
    calc.batched_relax_atoms(ats, fnames)

print(evals)
