"""Code to handle a digital twin data structure."""

import logging
import os
import sys

from uuid import uuid4

from ase import Atoms

import mp_api
from pymatgen.core.surface import SlabGenerator
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

from ocdata.core import Adsorbate, AdsorbateSlabConfig, Bulk, Slab

# sys.path.append("src")
# from structure_creation.materials_project_interface import mp_docs_from_symbols

logging.getLogger().setLevel(logging.INFO)

MP_API_KEY = os.environ["MP_API_KEY"]


class SlabDigitalTwin:
    """A class for a digital twin of a slab system."""

    dummy_adsorbate = Adsorbate(
        Atoms("H", positions=[[0, 0, 0]]), adsorbate_binding_indices=[0]
    )
    available_statuses = [
        "answer",
        "symbols",
        "bulk",
        "millers",
        "surface",
        "site_placement",
    ]

    def __init__(
        self,
        computational_object: object = None,
        computational_params: dict = {},
        info: dict = {},
        id=None,
        parent_twin_id=None,
    ):
        """Initialize self.

        Status indicates the most recent completed stage of creation
        (i.e. bulk, millers, surface,...). The computational object
        is the object underlying self."""
        self.computational_object = computational_object
        self.computational_params = computational_params
        self.info = info
        if id is None:
            self._id = uuid4()

    def copy(self):
        """Return a copy of self."""
        comp_obj = (
            self.computational_object.copy()
            if hasattr(self.computational_object, "copy")
            else self.computational_object
        )
        return SlabDigitalTwin(
            computational_object=comp_obj,
            computational_params=self.computational_params.copy(),
            info=self.info.copy(),
            parent_twin_id=self._id,
        )

    @property
    def row(self):
        """Return the database row associated with self."""
        row = {}
        for k in self.available_statuses:
            if k in self.computational_params:
                row[k] = self.computational_params[k]
            else:
                row[k] = None

    @property
    def status(self):
        """Return the curent state of creation."""
        max_idx = -1
        for i, k in enumerate(self.available_statuses):
            idx = self.available_statuses.index(k)
            if k in self.computational_params and idx > max_idx:
                max_idx = idx
        return self.available_statuses[max_idx]

    @property
    def completed(self):
        """Return whether creation is completed."""
        for k in self.available_statuses:
            if (
                k not in self.computational_params
                or self.computational_params[k] in None
            ):
                return False
        # else:
        return True

    def set_symbols(self, symbols: list[list[str]]):
        """Set the symbols for self, returning additional copies if needed."""
        if isinstance(symbols[0], str):
            symbols = [symbols]
        return_values = []
        for i, syms in enumerate(symbols):
            if i == 0:
                self.computational_params["symbols"] = syms
                self.computational_object = syms
            else:
                cpy = self.copy()
                cpy.set_symbols([syms])
                return_values.append(cpy)
        return return_values

    def get_bulks(self, filter_theoretical=True):
        """The the set of bulk available for self."""
        # Filter for materials with only the specified elements
        if self.status != "symbols":
            raise ValueError(
                "Cannot get available bulks without computing symbols, first."
            )

        with MPRester(MP_API_KEY) as mpr:
            docs = mpr.summary.search(elements=self.computational_params["symbols"])
        # Filter for materials with only the specified elements
        docs = [
            d
            for d in docs
            if all(
                [
                    str(elem) in self.computational_params["symbols"]
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
        return [d for d in sorted(docs, key=lambda d: d.formation_energy_per_atom)]

    def set_bulk(self, bulks: list[str]):
        """Set the bulk of self, returning copies if necessary.

        Bulks should be passed as a list of materials project docs.
        """
        if isinstance(bulks, str):
            bulks = [bulks]

        return_values = []
        for i, b in enumerate(bulks):
            if i == 0:
                self.computational_params["bulk"] = b.material_id
                self.computational_object = Bulk(
                    bulk_atoms=AseAtomsAdaptor().get_atoms(b.structure)
                )
            else:
                cpy = self.copy()
                cpy.set_bulk([b])
                return_values.append(cpy)

        return return_values

    #### No function for get Millers #### noqa:E266

    def set_millers(self, millers: list[tuple[int]]):
        """Set the miller indices given in the list, returning copies if necessary.

        Millers should be passed as a list of tuples of integers with length 3."""
        if isinstance(millers, tuple):
            millers = [millers]

        return_values = []
        for i, m in enumerate(millers):
            if i == 0:
                self.computational_params["millers"] = m
                self.computational_object = Slab.from_bulk_get_specific_millers(
                    m,
                    self.computational_object,
                    min_ab=8.0,  # consider reducing this before site placement?
                )
            else:
                cpy = self.copy()
                cpy.set_millers([m])
                return_values.append(cpy)
        return return_values

    def get_surfaces(self):
        """Get the possible surfaces for self."""
        return self.computational_object

    def set_surfaces(self, surfaces: list[Slab]):
        """Set the surfaces given in the list, returning copies if necessary.

        Surfaces should be lists of Slab objects."""
        if isinstance(surfaces, Slab):
            surfaces = [surfaces]

        return_values = []
        for i, s in enumerate(surfaces):
            if i == 0:
                self.computational_params["surface"] = (s.shift, s.top)
                self.computational_object = s
            else:
                cpy = self.copy()
                cpy.set_surfaces([s])
                return_values.append(cpy)
        return return_values

    def get_site_placements(self):
        """Get the binding sites associated with self."""
        adslab_config = AdsorbateSlabConfig(
            slab=self.computational_object,
            adsorbate=self.dummy_adsorbate,
            mode="heuristic",
        )
        return [tuple(s) for s in adslab_config.sites]

    def set_site_placement(self, binding_sites: list[tuple[float]]):
        """Set the binding sites given in the list, returning copies if necessary.

        Binding sites should be lists of Slab objects."""
        if isinstance(binding_sites, tuple):
            binding_sites = [binding_sites]

        return_values = []
        for i, site in binding_sites:
            if i == 0:
                self.computational_params["site_placement"] = site
                self.computational_object = (self.computational_object, site)
            else:
                cpy = self.copy()
                cpy.set_site_placement([site])
                return_values.append(cpy)
        return return_values
