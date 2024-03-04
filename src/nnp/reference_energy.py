"""Evaluate single point energies for xyz files."""

import logging
import pickle
import sys
import time

from pathlib import Path

from ase.io import Trajectory, write, ulm
import numpy as np
import pandas as pd

sys.path.append("src")
from nnp.oc import OCAdsorptionCalculator  # noqa:E402

logging.getLogger().setLevel(logging.INFO)


class OCReferenceCalculator:
    def __init__(self, reference_file: Path = Path("src", "nnp", "oc20_ref.pkl")):
        """Create self for the given path."""
        self.reference_path = reference_file
        with open(self.reference_path, "rb") as f:
            self.reference_data = pickle.load(f)

    def get_reference_energies(self, sid: int):
        """Get the reference energies for the given sid"""
        key = f"random{sid}"
        reference_energies = self.reference_data[key]
        return reference_energies


total_time = 0
data_path = Path("src/nnp/3_1_24_relaxations")

calc = OCAdsorptionCalculator(
    **{
        "model": "gemnet-oc-22",
        "traj_dir": data_path,
        "batch_size": 75,
        "device": "cuda",
        "ads_tag": 2,
        "fmax": 0.05,
        "steps": 0,
    }
)
results = pd.DataFrame()
batch_size = 75
batch = []
reference_energies = {}
bulk_slabs = []
for traj in data_path.rglob("*.traj"):
    try:
        bulk = traj.stem.split("bulk_")[-1].split("_")[0]
        slab = traj.stem.split("slab_")[-1].split("_")[0]
        bulk_slab = (bulk, slab)

        if bulk_slab not in reference_energies:
            ats = Trajectory(str(traj))[0]
            bulk_slabs.append(bulk_slab)
            batch.append(ats)
            reference_energies[bulk_slab] = None

    except ulm.InvalidULMFileError:
        logging.warning(f"Could not read file {str(traj)}.")
    except ValueError as err:
        if "buffer is smaller than requested size" in str(err):
            logging.warning(f"Could not read file {str(traj)}, with error {str(err)}.")
        else:
            raise err

    if len(batch) == batch_size:
        logging.info("==== Running Batch ====")
        start = time.time()
        evaled_ats = calc.calculate_slab_energies(batch)
        end = time.time()
        total_time += end - start
        logging.info((end - start))

        for i, ats in enumerate(evaled_ats):
            reference_energies[bulk_slabs[i]] = ats.get_potential_energy()
            write(
                str(
                    data_path
                    / f"reference_energy_bulk_{bulk_slabs[i][0]}_slab_{bulk_slabs[i][1]}.xyz"
                ),
                ats,
            )

        batch = []
        bulk_slabs = []

        logging.info(end - start)


if len(batch) > 0:
    logging.info("==== Running Final Batch ====")
    start = time.time()
    evaled_ats = calc.calculate_slab_energies(batch)
    end = time.time()
    total_time += end - start
    logging.info((end - start))

    for i, ats in enumerate(evaled_ats):
        reference_energies[bulk_slabs[i]] = ats.get_potential_energy()
        write(
            str(
                data_path
                / f"reference_energy_bulk_{bulk_slabs[i][0]}_slab_{bulk_slabs[i][1]}.xyz"
            ),
            ats,
        )

logging.info(f"Total time: {total_time}.")

pandas_data_dict = [
    {"bulk": str(k[0]), "slab": str(k[1]), "reference_energy": v}
    for k, v in reference_energies.items()
]
df = pd.DataFrame(pandas_data_dict)
df.to_csv(data_path / "reference_calculations.csv")
logging.info(
    f"Saved reference energies to {str(data_path / 'reference_calculations.csv')}."
)
