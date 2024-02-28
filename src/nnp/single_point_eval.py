"""Evaluate single point energies for xyz files."""

import logging
import pickle
import sys
import time

from pathlib import Path

from ase.io import read
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


for model in ["gemnet-t"]:  # :["gemnet-oc-22", "gemnet-oc-large", "gemnet-t"]:
    total_time = 0
    data_path = Path("src/nnp/methanol_chemreasoner_results")

    calc = OCAdsorptionCalculator(
        **{
            "model": model,
            "traj_dir": data_path,
            "batch_size": 40,
            "device": "cuda",
            "ads_tag": 2,
            "fmax": 0.05,
            "steps": 64,
        }
    )
    results = pd.DataFrame()
    batch_size = 42
    batch = []
    bulk_elements = []
    ads_elements = []
    oc_energies = []
    fnames = []
    for xyz in data_path.rglob("*.xyz"):
        print(xyz)
        fnames.append(str(xyz))
        # traj_path = str(xyz).replace(str(data_path), "")[1:]
        ats = read(str(xyz))
        oc_energies.append(ats.get_potential_energy())
        bulk_elements.append(np.unique(ats.numbers[ats.get_tags() < 2]).tolist())
        ads_elements.append(np.unique(ats.numbers[ats.get_tags() == 2]).tolist())
        batch.append(ats)
        if len(batch) == 1:
            evaled_ats = calc.static_eval(batch)
            start = time.time()
            evaled_ats = calc.static_eval(batch)
            end = time.time()
            total_time += end - start
            logging.info((end - start) / 1)
            energies = [ats.get_potential_energy() for ats in evaled_ats]
            data = {
                "file": fnames,
                "bulk_elements": bulk_elements,
                "ads_elements": ads_elements,
                "gnn_energies": energies,
                "oc_energies": oc_energies,
            }
            results = pd.concat([results, pd.DataFrame(data)])
            batch = []
            fnames = []
            bulk_elements = []
            ads_elements = []
            oc_energies = []

            logging.info(end - start)

    if len(batch) > 0:
        evaled_ats = calc.static_eval(batch.copy())
        start = time.time()
        for i in range(1):
            evaled_ats = calc.static_eval(batch.copy())
        end = time.time()
        total_time += end - start
        energies = [ats.get_potential_energy() for ats in evaled_ats]
        data = {
            "file": fnames,
            "bulk_elements": bulk_elements,
            "ads_elements": ads_elements,
            "gnn_energies": energies,
            "oc_energies": oc_energies,
        }
        results = pd.concat([results, pd.DataFrame(data)])

        logging.info((end - start) / 1)
logging.info(f"Total time: {total_time}.")
# results.to_csv(f"{model}_gnn_single_point.csv")
# data_path = Path("src/nnp/oc_eval_set")

# for xyz in data_path.rglob("*.xyz"):
#     ats = read(str(xyz))
#     ref_calc = OCReferenceCalculator()
#     print(ref_calc.get_reference_energies(ats.info["sid"]))
