"""Evaluate relaxed adsorption eneries for the given xyz files."""

import logging
import random
import sys
import time


from pathlib import Path

from ase.io import read
import pandas as pd

sys.path.append("src")
from nnp.oc import OCAdsorptionCalculator  # noqa:E402

logging.getLogger().setLevel(logging.INFO)

data_path = Path("src/nnp/3_1_24_relaxations")

calc = OCAdsorptionCalculator(
    **{
        "model": "gemnet-oc-22",
        "traj_dir": data_path,
        "batch_size": 30,
        "device": "cpu",
        "ads_tag": 2,
        "fmax": 0.05,
        "steps": 100,
    }
)

batch_size = 64
batch = []
fnames = []
evals = []
for xyz in data_path.rglob("*.xyz"):
    print(xyz)
    traj_path = str(xyz).replace(str(data_path), "")[1:]
    p = data_path / (traj_path.replace(".xyz", "") + ".traj")
    p_tmp = data_path / (traj_path.replace(".xyz", "") + ".traj_tmp")

    if not p.exists() and not p_tmp.exists():
        # Append data to batch
        fnames.append(traj_path.replace(".xyz", ""))
        print(data_path / traj_path)
        ats = read(str(xyz))
        batch.append(ats)
    else:
        # Just keep going
        print(f"Skipping {str(p)}.")

    if len(batch) == batch_size:
        logging.info("=== Running Batch ===")
        start = time.time()
        calc.batched_relax_atoms(batch, fnames)
        end = time.time()
        logging.info(f"TIMING: One batch {end - start}.")
        batch = []
        fnames = []

if len(batch) > 0:
    logging.info("=== Running Final Batch ===")
    calc.batched_relax_atoms(batch, fnames)
    start = time.time()
    calc.batched_relax_atoms(batch, fnames)
    end = time.time()
    logging.info(f"TIMING: One batch {end - start}.")

print(evals)
