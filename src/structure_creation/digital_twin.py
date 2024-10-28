"""Code to handle a digital twin data structure."""

import logging
import math
import os
import sys

from ast import literal_eval
from copy import deepcopy
from typing import Union
from uuid import uuid4

from ase import Atoms

import mp_api
from pymatgen.core.surface import SlabGenerator
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

import numpy as np

from ocdata.core import Adsorbate, AdsorbateSlabConfig, Bulk, Slab

# sys.path.append("src")
# from structure_creation.materials_project_interface import mp_docs_from_symbols

logging.getLogger().setLevel(logging.INFO)

MP_API_KEY = os.environ["MP_API_KEY"]

retrieved_mp_docs = {}


class CatalystDigitalTwin:
    """A class for a digital twin of a slab system."""

    _collected_mp_ids = {}
    _collected_slabs = {}

    dummy_adsorbate = Adsorbate(
        Atoms("H", positions=[[0, 0, 0]]), adsorbate_binding_indices=[0]
    )
    available_statuses = [
        "llm_answer",
        "symbols",
        "bulk",
        "millers",
        "surface",
        "site_placement",
    ]

    def __init__(
        self,
        computational_objects: object = None,
        computational_params: dict = None,
        info: dict = None,
        _id=None,
        children_ids: list = None,
        canceled=False,
        reward=None,
    ):
        """Initialize self.

        Status indicates the most recent completed stage of creation
        (i.e. bulk, millers, surface,...). The computational object
        is the object underlying self."""

        self.computational_objects = (
            computational_objects if computational_objects is not None else {}
        )
        self.computational_params = (
            computational_params if computational_params is not None else {}
        )
        self.info = info if info is not None else {}
        if _id is None:
            self._id = str(uuid4())
        self.children_ids = children_ids if children_ids is not None else []
        self.canceled = canceled
        self.reward = reward
        self._site_placements = None

    def set_reward(self, reward: float):
        """Set the value of the reward."""
        self.reward = reward

    def get_reward(self):
        """Return the reward associated with self for given reaction info."""
        return self.reward

    def set_uncertainty(self, uncertainty: float):
        """Set the value of the reward."""
        self.uncertainty = uncertainty

    def get_uncertainty(self):
        """Return the reward associated with self for given reaction info."""
        return self.uncertainty

    def update_info(self, status_key: str, updates: dict, start_new: bool = False):
        """Update the info for the self with updates."""
        if status_key not in self.info:
            self.info[status_key] = [updates]
        elif start_new:
            self.info[status_key].append(updates)
        else:
            self.info[status_key][-1].update(updates)

    def copy(self, copy_info: bool = False):
        """Return a copy of self."""
        if copy_info:
            info = deepcopy(self.info.copy())
        else:
            info = {}
        return CatalystDigitalTwin(
            computational_objects=deepcopy(self.computational_objects),
            computational_params=deepcopy(self.computational_params),
            info=info,
        )

    @classmethod
    def from_row(cls, row_data):
        """Return CatalystDigitalTwin from the given row data."""
        start_dt = cls()
        dt = start_dt
        for k in start_dt.available_statuses:
            if row_data[k] is not None and not (
                isinstance(row_data[k], float) and math.isnan(row_data[k])
            ):

                method = getattr(dt, f"set_{k}s", None)
                if method is None:
                    try:
                        method = getattr(dt, f"set_{k}")
                    except AttributeError:
                        raise AttributeError(
                            f"Catalyst digital twin does not have method {f'set_{k}s'} or {f'set_{k}'}."
                        )
                dt = method([row_data[k]])[0]
        dt._id = row_data["id"]
        return dt

    def return_row(self, metadata: bool = False):  # TODO: Do something with LLM queries
        """Return the data stored within the digital twin."""
        row = deepcopy(self.computational_params)
        row["id"] = self._id
        row.update({k: None for k in self.available_statuses if k not in row})
        if metadata:
            return row, deepcopy(self.info)
        return row

    def return_slab(self):
        """Return the slab associated with self."""
        slab, _ = self.computational_objects["site_placement"]
        return slab.atoms

    def return_adslab_config(
        self, adsorbate: Adsorbate, num_augmentations_per_site: int = 1
    ) -> AdsorbateSlabConfig:
        """Get the adsorbate+slab configuration specified by self."""
        slab, site = self.computational_objects["site_placement"]
        adslab_config = AdsorbateSlabConfig(
            slab=slab,
            adsorbate=adsorbate,
            num_sites=1,
            num_augmentations_per_site=num_augmentations_per_site,
            mode="random",
        )
        adslab_config.sites = [np.array(site)]
        adslab_config.atoms_list, adslab_config.metadata_list = (
            adslab_config.place_adsorbate_on_sites(
                adslab_config.sites,
                adslab_config.num_augmentations_per_site,
                adslab_config.interstitial_gap,
            )
        )
        for ats in adslab_config.atoms_list:
            force_equal_length_arrays(ats)
        return adslab_config

    @property
    def status(self):
        """Return the curent state of creation."""
        max_idx = -1
        for i, k in enumerate(self.available_statuses):
            idx = self.available_statuses.index(k)
            if k in self.computational_params and idx > max_idx:
                max_idx = idx
        if max_idx == -1:
            return None
        else:
            return self.available_statuses[max_idx]

    @property
    def completed(self):
        """Return whether creation is completed."""
        return self.status == self.available_statuses[-1]

    def set_llm_answers(self, answers: list[str]):
        """Set the answer for self, returning additional copies if needed."""
        if isinstance(answers, str):
            answers = [answers]
        return_values = []
        for i, ans in enumerate(answers):
            cpy = self.copy()
            cpy.computational_params["llm_answer"] = ans
            cpy.computational_objects["llm_answer"] = ans
            return_values.append(cpy)
        return return_values

    def set_symbols(self, symbols: list[list[str]]):
        """Set the symbols for self, returning additional copies if needed."""
        if isinstance(symbols[0], str):
            symbols = [symbols]
        return_values = []
        for i, syms in enumerate(symbols):
            if isinstance(syms, str):
                syms = literal_eval(syms)
            cpy = self.copy()
            cpy.computational_params["symbols"] = syms
            cpy.computational_objects["symbols"] = syms
            return_values.append(cpy)
        return return_values

    @classmethod
    def get_bulks_id(cls, mp_ids):
        """Get the bulks associated with the given mp_ids."""
        docs = [
            (
                cls._collected_mp_ids[mp_id]
                if mp_id in cls._collected_mp_ids.keys()
                else None
            )
            for mp_id in mp_ids
        ]
        if any([d is None for d in docs]):
            with MPRester(MP_API_KEY) as mpr:
                new_docs = mpr.summary.search(
                    material_ids=[
                        mp_id
                        for mp_id in mp_ids
                        if mp_id not in cls._collected_mp_ids.keys()
                    ]
                )
                i, j = 0, 0
                while i < len(docs):
                    if docs[i] is None:
                        docs[i] = new_docs[j]
                        j += 1
                    i += 1

            cls._collected_mp_ids.update(
                {mp_id: doc for mp_id, doc in zip(mp_ids, docs)}
            )
        return docs

    def get_bulks(self, filter_theoretical=False):
        """The the set of bulk available for self."""
        # Filter for materials with only the specified elements
        if not hasattr(self, "_bulks"):
            with MPRester(MP_API_KEY) as mpr:
                docs = mpr.summary.search(
                    elements=self.computational_objects["symbols"]
                )
            # Filter for materials with only the specified elements
            docs = [
                d
                for d in docs
                if all(
                    [
                        str(elem) in self.computational_objects["symbols"]
                        for elem in d.elements
                    ]
                )
            ]
            # Filter for materials that are experimentally verified
            if filter_theoretical:
                docs_filtered = [d for d in docs if not d.theoretical]
                if len(docs_filtered) == 0:
                    logging.warning(
                        "Unable to filter for theoretical since no experimentally verified materials exist."
                    )
                else:
                    docs = docs_filtered
            self._bulks = [
                d for d in sorted(docs, key=lambda d: d.formation_energy_per_atom)
            ]
        return self._bulks

    def set_bulk(self, bulks: list[str]):
        """Set the bulk of self, returning copies if necessary.

        Bulks should be passed as a list of materials project docs.
        """
        if isinstance(bulks, str):
            bulks = [bulks]

        mp_id_idxs = [i for i in range(len(bulks)) if isinstance(bulks[i], str)]
        if len(mp_id_idxs) > 0:
            docs = self.get_bulks_id([bulks[i] for i in mp_id_idxs])
            for i, j in enumerate(mp_id_idxs):
                bulks[j] = docs[i]

        return_values = []
        for i, b in enumerate(bulks):
            cpy = self.copy()
            cpy.computational_params["bulk"] = b.material_id
            cpy.computational_objects["bulk"] = b
            return_values.append(cpy)

        return return_values

    #### No function for get Millers #### noqa:E266

    @classmethod
    def fetch_slab_cache(cls, mp_id: str, millers: tuple):
        """Return the calculated slabs for the given bulk+millers."""
        return cls._collected_slabs.get(f"{mp_id}+{millers}", None)

    @classmethod
    def update_slab_cache(cls, mp_id: str, millers: tuple, slabs: list):
        """Return the calculated slabs for the given bulk+millers."""
        return cls._collected_slabs.update({f"{mp_id}+{millers}": slabs})

    def set_millers(self, millers: list[tuple[int]]):
        """Set the miller indices given in the list, returning copies if necessary.

        Millers should be passed as a list of tuples of integers with length 3. If
        millers are in Miller-Bravais format, this function will automatically convert
        to millers."""
        if isinstance(millers, tuple):
            millers = [millers]

        return_values = []
        for m in millers:
            if isinstance(m, int):
                # convert to tuple, i.e. 111 -> (1,1,1)
                m = tuple([int(a) for a in list(str(m))])
            if isinstance(m, str):
                m = literal_eval(m)
            if len(m) != 3:
                m = convert_miller_bravais_to_miller(m)
            cpy = self.copy()
            cpy.computational_params["millers"] = m
            slabs = self.fetch_slab_cache(
                cpy.computational_params["bulk"], cpy.computational_params["millers"]
            )
            cpy.computational_objects["millers"] = (
                slabs
                if slabs is not None
                else Slab.from_bulk_get_specific_millers(
                    m,
                    Bulk(
                        bulk_atoms=AseAtomsAdaptor().get_atoms(
                            cpy.computational_objects["bulk"].structure
                        )
                    ),
                    min_ab=8.0,  # TODO: consider reducing this before site placement?
                )
            )
            self.update_slab_cache(
                cpy.computational_params["bulk"],
                cpy.computational_params["millers"],
                cpy.computational_objects["millers"],
            )
            return_values.append(cpy)
        return return_values

    def get_surfaces(self):
        """Get the possible surfaces for self."""
        return self.computational_objects["millers"]

    def set_surfaces(self, surfaces: list[Slab]):
        """Set the surfaces given in the list, returning copies if necessary.

        Surfaces should be lists of Slab objects."""
        if isinstance(surfaces, Slab):
            surfaces = [surfaces]

        return_values = []
        for s in surfaces:
            if isinstance(s, str):
                s = literal_eval(s)
            if isinstance(s, tuple):
                s = min(
                    self.computational_objects["millers"],
                    key=lambda slab: abs(s[0] - slab.shift)
                    + (np.inf if s[1] != slab.top else 0),  # Find the closest surface
                )
            cpy = self.copy()
            cpy.computational_params["surface"] = (s.shift, s.top)
            cpy.computational_objects["surface"] = s
            return_values.append(cpy)
        return return_values

    def get_site_placements(self):
        """Get the binding sites associated with self."""
        if self._site_placements is None:
            adslab_config = AdsorbateSlabConfig(
                slab=self.computational_objects["surface"],
                adsorbate=self.dummy_adsorbate,
                mode="heuristic",
            )
            self._site_placements = [tuple(s) for s in adslab_config.sites]
        return self._site_placements

    def set_site_placements(self, binding_sites: list[tuple[float]]):
        """Set the binding sites given in the list, returning copies if necessary.

        Binding sites should be lists of Slab objects."""
        if isinstance(binding_sites, tuple):
            binding_sites = [binding_sites]
        if isinstance(binding_sites[0], float):
            binding_sites = [binding_sites]

        return_values = []
        for site in binding_sites:
            if isinstance(site, str):
                site = literal_eval(site)
            cpy = self.copy()
            cpy.computational_params["site_placement"] = site
            cpy.computational_objects["site_placement"] = (
                cpy.computational_objects["surface"],
                site,
            )
            return_values.append(cpy)
        return return_values


def convert_miller_bravais_to_miller(miller_bravais_indices: tuple[int]):
    """Convert a 4-tuple Miller-Bravais indices to 3-tuple Miller indices for surface plane.

    Uses the definitions from:
    https://ssd.phys.strath.ac.uk/resources/crystallography/crystallographic-direction-calculator/

    Coordinate transformation hkil -> hkl defined by:

    i = -(h+k)
    h = h
    k = k
    l = l

    This function simply drops the third entry in miller_bravais_indices.
    """
    assert len(miller_bravais_indices) == 4
    assert miller_bravais_indices[2] == -(
        miller_bravais_indices[0] + miller_bravais_indices[1]
    ), "Invalid Miller-Bravais lattice."
    return (
        miller_bravais_indices[0],
        miller_bravais_indices[1],
        miller_bravais_indices[3],
    )


def force_equal_length_arrays(ats: Atoms):
    """Force the arrays in the atoms object and make sure they are of equa length."""
    longest_arrays = []
    longest_arrays_length = 0
    for k, arr in ats.arrays.items():
        if arr.shape[0] == longest_arrays_length:
            longest_arrays.append(k)
        elif arr.shape[0] > longest_arrays_length:
            for old_k in longest_arrays:
                ats.arrays.pop(old_k)
            longest_arrays = [k]
            longest_arrays_length = arr.shape[0]
        else:
            ats.arrays.pop(k)
