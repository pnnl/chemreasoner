"""Script to test GemNet on the OC dataset."""

import json
import logging
import pickle
import sys

from pathlib import Path

from ase import Atoms
from ase.io import read

import pandas as pd

sys.path.append("src")
from nnp.oc import OCAdsorptionCalculator

logging.getLogger().setLevel(logging.INFO)


data_path = Path("microstructure_planner_test")

calc = OCAdsorptionCalculator(
    **{
        "model": "gemnet-oc-22",
        "traj_dir": data_path,
        "batch_size": 80,
        "device": "cuda",
        "ads_tag": 2,
        "fmax": 0.03,
        "steps": 200,
    }
)
atoms = {}
for p in data_path.rglob("*.xyz"):
    atoms[p.stem] = read(str(p))

# print(8 not in [atom.number for atom in atoms['oc_20/0_23101'] if (atom.tag == 0)])

timing = {}
energies = {}
keys = list(atoms.keys())

atoms_list = []
atoms_names = []
for k, v in atoms.items():
    atoms_list.append(v)
    name = Path("trajectories_e_tot") / k
    atoms_names.append(str(name))
    (data_path / name).parent.mkdir(parents=True, exist_ok=True)

start_timing = calc.gnn_time
relaxed_atoms = calc.batched_relax_atoms(atoms_list, atoms_names=atoms_names)
end_timing = calc.gnn_time
timing["total_energy"] = end_timing - start_timing

for k, ats in zip(keys, relaxed_atoms):
    energies[k] = {"relaxed_energy": ats.get_potential_energy()}

# Run slab reference calculation

bulk_atoms = {}
for k, ats in atoms.items():
    energies[k].update({"adsorbate_reference_energy": 0})
    bulk_ats = Atoms()
    e_ref = 0
    for i, at in enumerate(ats):
        if at.tag == 2:  # part of the adsorbate
            energies[k]["adsorbate_reference_energy"] += calc.ads_references[
                ats.get_atomic_numbers()[i]
            ]
        else:
            bulk_ats.append(at)

    tags_mask = ats.get_tags() != 2

    for key, value in ats.arrays.items():
        bulk_ats.arrays[key] = value[tags_mask]

    bulk_ats.set_pbc(ats.get_pbc())
    bulk_ats.set_cell(ats.get_cell())
    bulk_atoms[k] = bulk_ats.copy()

bulk_atoms_list = []
bulk_atoms_names = []
for k, v in bulk_atoms.items():
    atoms_list.append(v)
    name = Path("trajectories_e_slab") / k
    bulk_atoms_names.append(str(name))
    (data_path / name).parent.mkdir(parents=True, exist_ok=True)

start_timing = calc.gnn_time
bulk_relaxed_atoms = calc.batched_relax_atoms(
    bulk_atoms_list, atoms_names=bulk_atoms_names
)
end_timing = calc.gnn_time

timing["slab_energy"] = end_timing - start_timing

for k, ats in zip(keys, bulk_relaxed_atoms):
    energies[k].update({"slab_relaxed_energy": ats.get_potential_energy()})


# Run slab prime reference calculation

bulk_atoms_prime = {}
for k, ats in zip(keys, relaxed_atoms):
    bulk_ats = Atoms()
    e_ref = 0
    for i, t in enumerate(ats.get_tags()):
        if t != 2:  # part of the bulk
            bulk_ats.append(ats[i])

    tags_mask = ats.get_tags() != 2

    for key, value in ats.arrays.items():
        bulk_ats.arrays[key] = value[tags_mask]

    bulk_ats.set_pbc(ats.get_pbc())
    bulk_ats.set_cell(ats.get_cell())
    bulk_atoms_prime[k] = bulk_ats.copy()


bulk_atoms_prime_list = []
bulk_atoms_prime_names = []
for k, v in bulk_atoms_prime.items():
    atoms_list.append(v)
    name = Path("trajectories_e_slab_prime") / k
    bulk_atoms_prime_names.append(str(name))
    (data_path / name).parent.mkdir(parents=True, exist_ok=True)

start_timing = calc.gnn_time
bulk_relaxed_atoms_prime = calc.batched_relax_atoms(
    bulk_atoms_prime_list, atoms_names=bulk_atoms_prime_names
)
end_timing = calc.gnn_time
timing["slab_prime_energy"] = end_timing - start_timing

for k, ats in zip(keys, bulk_relaxed_atoms_prime):
    energies[k].update({"slab_relaxed_prime_energy": ats.get_potential_energy()})


# Write the final results to disk
df = []
for k, v in energies.items():
    data = {"key": k}
    data.update(v)
    df.append(data)

pd.DataFrame(df).to_csv(str(data_path / "energy_results.csv"), index=False)
with open(data_path / "energy_timing_results.json", "w") as f:
    json.dump(timing, f)
