"""Evaluate relaxed adsorption eneries for the given xyz files."""

import logging
import sys
import time


from pathlib import Path

from ase.io import read
import pandas as pd

sys.path.append("src")
from nnp.oc import OCAdsorptionCalculator  # noqa:E402

logging.getLogger().setLevel(logging.INFO)

data_path = Path("src/nnp/adslabs_CO2_test")

calc = OCAdsorptionCalculator(
    **{
        "model": "gemnet-t",
        "traj_dir": data_path,
        "batch_size": 200,
        "device": "cuda",
        "ads_tag": 2,
        "fmax": 0.05,
        "steps": 300,
    }
)

batch_size = 300
batch = []
fnames = []
evals = []
for xyz in sorted(data_path.rglob("*.xyz")):
    print(xyz)
    traj_path = str(xyz).replace(str(data_path), "")[1:]
    p = data_path / (traj_path.replace(".xyz", "") + ".traj")

    if not p.exists():
        # Append data to batch
        fnames.append(traj_path.replace(".xyz", ""))
        print(data_path / traj_path)
        ats = read(str(xyz))
        batch.append(ats)
    else:
        # Just keep going
        print(f"Skipping {str(p)}.")

    if len(batch) == batch_size:
        print("=== Running Batch ===")
        start = time.time()
        calc.batched_relax_atoms(batch, fnames)
        end = time.time()
        logging.info(start - end)
        batch = []
        fnames = []

if len(batch) > 0:
    print("=== Running Final Batch ===")
    calc.batched_relax_atoms(batch, fnames)

print(evals)
