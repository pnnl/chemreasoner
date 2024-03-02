"""Interface for generating adlsabs from the materials project database."""

import logging
import os
import pickle
import sys
import time

from multiprocessing import Pool
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

import mp_api

from ase import Atoms
from ase.data import chemical_symbols
from ase.io import write
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

from ocdata.core import Adsorbate, AdsorbateSlabConfig, Bulk, Slab

sys.path.append("src")
from llm.ase_interface import ads_symbols_to_structure

logging.getLogger().setLevel(logging.INFO)

MP_API_KEY = os.getenv("MP_API_KEY")

with open(
    Path("ext", "Open-Catalyst-Dataset", "ocdata", "databases", "pkls", "bulks.pkl"),
    "rb",
) as f:
    _bulk_db = pickle.load(f)

_bulk_df = pd.DataFrame(_bulk_db)


def mp_docs_from_symbols(syms: list[str]) -> list:
    time1 = time.time()
    with MPRester(MP_API_KEY) as mpr:
        time2 = time.time()
        print(f"Time to start client: {time2 - time1}.")
        docs = mpr.materials.summary.search(elements=syms)
        time3 = time.time()
        print(f"Time to get structures: {time3 - time2}.")
    docs = [d for d in docs if all([str(elem) in syms for elem in d.elements])]
    return docs


def retrive_materials_project_structures(mp_ids: list[str]) -> list:
    time1 = time.time()
    with MPRester(MP_API_KEY) as mpr:
        time2 = time.time()
        print(f"Time to start client: {time2 - time1}.")
        docs = mpr.materials.summary.search(material_ids=mp_ids)
        print(docs)
        time3 = time.time()
        print(f"Time to get structures: {time3 - time2}.")

    return docs


def ocp_bulks_from_mp_ids(mp_ids: list) -> list[Union[Bulk, None]]:
    """Return a list of ocp Bulk objects from list of materials project docs.

    Unfound bulks are returned in the list as "None".
    """
    oc_bulks = []
    leftover_mp_ids = []
    idxs = []
    for i, mp_id in enumerate(mp_ids):
        if any(_bulk_df["src_id"] == mp_id):
            b = Bulk(bulk_src_id_from_db=mp_id, bulk_db=_bulk_db)
            b.atoms.info.update({"src_id": mp_id})
            oc_bulks.append(b)
        else:
            leftover_mp_ids.append(mp_id)
            idxs.append(i)
            oc_bulks.append(None)

    # Go back and fill in locally missing entries
    if len(leftover_mp_ids) > 0:
        docs = retrive_materials_project_structures(leftover_mp_ids)
        for i, doc in zip(idxs, docs):
            oc_bulks[i] = Bulk(bulk_atoms=AseAtomsAdaptor().get_atoms(doc.structure))

    return oc_bulks


def ocp_bulks_from_symbols(syms: list[str]) -> list[Bulk]:
    """Get the ocp bulk from the given list of symbols."""
    docs = mp_docs_from_symbols(syms)
    return [
        b for b in ocp_bulks_from_mp_ids([d.material_id for d in docs]) if b is not None
    ]


def create_adslab_config(adslab_pair: list[Slab, Adsorbate]) -> Atoms:
    """Return the list of adslabs for the given sla, which the given adsorbate."""
    atoms_list = AdsorbateSlabConfig(
        adslab_pair[0],
        adsorbate=adslab_pair[1],
        num_augmentations_per_site=1,
        mode="heuristic",
    ).atoms_list
    for ats in atoms_list:
        if "bulk_wyckoff" in ats.arrays.keys():
            ats.arrays["bulk_wyckoff"] = np.array(
                [val if val != "" else "N/A" for val in ats.arrays["bulk_wyckoff"]]
            )

    return atoms_list


def get_all_adslabs(
    bulk: Bulk,
    adsorbate: Adsorbate,
):
    """Get all the adslabs for the given bulk."""
    slabs = bulk.get_slabs()
    adslab_pairs = [[s, adsorbate] for s in slabs]
    return [create_adslab_config(p) for p in adslab_pairs]


def prepare_ocp_adsorbate(
    adsorbate: Union[Adsorbate, Atoms], adsorbate_binding_indices: list[int] = None
):
    """Create ocp_adsorption object from given adsorbate."""
    if isinstance(adsorbate, Atoms):
        adsorbate_binding_indices = (
            adsorbate_binding_indices if adsorbate_binding_indices is not None else [0]
        )
        ads_obj = (
            adsorbate
            if isinstance(adsorbate, Adsorbate)
            else Adsorbate(
                adsorbate_atoms=adsorbate,
                adsorbate_binding_indices=adsorbate_binding_indices,
            )
        )
    else:
        if adsorbate_binding_indices is not None:
            logging.warning("Ignoring given adsorbate binding indices.")
        ads_obj = adsorbate
    return ads_obj


def ocp_adslabs_from_symbols(
    syms: list[str],
    adsorbate: Union[Adsorbate, Atoms],
    adsorbate_binding_indices: list[int] = None,
):
    """Return all possible adslabes for bulkds specified by syms adsorbate."""
    ads_obj = prepare_ocp_adsorbate(
        adsorbate=adsorbate,
        adsorbate_binding_indices=adsorbate_binding_indices,
    )

    bulks = ocp_bulks_from_symbols(syms)
    print(len(bulks))
    structure_list = []
    for b in bulks:
        structure_list.append(get_all_adslabs(b, ads_obj))

    return structure_list


def ocp_adslabs_from_mp_ids(
    mp_ids: list[str],
    adsorbate: Union[Adsorbate, Atoms],
    adsorbate_binding_indices: list[int] = None,
):
    """Return all possible adslabes for bulkds specified by syms adsorbate."""
    ads_obj = prepare_ocp_adsorbate(
        adsorbate=adsorbate,
        adsorbate_binding_indices=adsorbate_binding_indices,
    )
    bulks = ocp_bulks_from_mp_ids(mp_ids)
    structure_list = []
    for i, b in enumerate(bulks):
        if b is not None:
            structure_list.append(get_all_adslabs(b, ads_obj))
        else:
            logging.warning(
                f"Skipping bulk {mp_ids[i]} since it was "
                "not found in the local database."
            )

    return structure_list


def save_adslabs(
    atoms_list: list[Atoms],
    atoms_names: list[str],
    save_dir: Path,
    create_dir: bool = False,
):
    """Save the given atoms in the directory listed."""
    if create_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for ats, fname in zip(atoms_list, atoms_names):
        write(str(save_dir / fname), ats)


def _sample_adslabs(ads):
    adsorbate_binding_indices = {"CO2": [2], "*CO": [0], "*OCHO": [0]}
    ads_ats = ads_symbols_to_structure(ads)
    ads_obj = Adsorbate(
        adsorbate_atoms=ads_ats,
        adsorbate_binding_indices=adsorbate_binding_indices[ads],
    )
    materials = [
        "mp-30",
        "mp-1017539",
        "mp-1046785",
        # "mp-1047156",
        # "mp-1351757",
    ]
    start = time.time()
    adslabs = ocp_adslabs_from_mp_ids(materials, ads_obj)
    for b_id in range(len(adslabs)):
        savedir = Path("3_1_24_relaxations", f"{ads}_{materials[b_id]}")
        for s_id in range(len(adslabs[b_id])):
            for a_id in range(len(adslabs[b_id][s_id])):
                savedir.mkdir(parents=True, exist_ok=True)
                # print(adslabs[b_id][s_id][a_id].info)
                path = savedir / f"bulk_{materials[b_id]}_slab_{s_id}_ads_{a_id}.xyz"
                write(str(path), adslabs[b_id][s_id][a_id])
    end = time.time()
    return end - start


if __name__ == "__main__":
    # adsorbate_binding_indices = {"CO2": [2], "*CO": [0], "*OCHO": [0]}
    adsorbate_times = {}
    pool = Pool(3)
    print(["CO2", "*CO", "*OCHO"])
    print(pool.map(_sample_adslabs, ["CO2", "*CO", "*OCHO"]))

    # for ads in ["CO2", "*CO", "*OCHO"]:
    #     ads_ats = ads_symbols_to_structure(ads)
    #     ads_obj = Adsorbate(
    #         adsorbate_atoms=ads_ats,
    #         adsorbate_binding_indices=adsorbate_binding_indices[ads],
    #     )
    #     materials = [
    #         "mp-30",
    #         "mp-1017539",
    #         "mp-1046785",
    #         # "mp-1047156",
    #         # "mp-1351757",
    #     ]
    #     start = time.time()
    #     adslabs = ocp_adslabs_from_mp_ids(materials, ads_obj)
    #     for b_id in range(len(adslabs)):
    #         savedir = Path(f"{ads}_{materials[b_id]}")
    #         for s_id in range(len(adslabs[b_id])):
    #             for a_id in range(len(adslabs[b_id][s_id])):
    #                 savedir.mkdir(parents=True, exist_ok=True)
    #                 # print(adslabs[b_id][s_id][a_id].info)
    #                 path = (
    #                     savedir / f"bulk_{materials[b_id]}_slab_{s_id}_ads_{a_id}.xyz"
    #                 )
    #                 write(str(path), adslabs[b_id][s_id][a_id])
    #     end = time.time()
    #     adsorbate_times[ads] = end - start
    print(adsorbate_times)
