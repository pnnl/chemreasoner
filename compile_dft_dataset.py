"""Compile dataset for UQ test.

This code is modified from the Open Catalyst Project scripts.
"""

import pickle

from pathlib import Path
from tqdm import tqdm

from ase.io import read
import lmdb

import torch

from ocpmodels.datasets.lmdb_dataset import LmdbDataset
from ocpmodels.preprocessing import AtomsToGraphs

a2g = AtomsToGraphs(
    max_neigh=50,
    radius=6,
    r_energy=True,
    r_forces=True,
    r_fixed=True,
    r_distances=False,
    r_pbc=True,
)

dft_adsorption_energies = {
    "CuZn_CO2": -0.066,
    "CuZn_CHOH": 5.951,
    "CuZn_OCHO": 5.836,
    "CuZn_OHCH3": 2.699,
    "CuAlZn_CO2": 6.816,
    "CuAlZn_CHOH": -1.824,
    "CuAlZn_OCHO": 2.820,
    "CuAlZn_OHCH3": -5.615,
}
lmdb_dir = Path("test_lmdb")
i = 0
with lmdb.open(
    str(Path("test_lmdb", "00.lmdb")),
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
) as db:
    for _, p in tqdm(
        enumerate(
            Path("/Users/spru445/Desktop/methanol_chemreasoner_results").rglob("*.xyz")
        )
    ):

        print(p)
        fname = p.stem
        ats = read(str(p))

        data_object = a2g.convert(ats)
        # add atom tags
        data_object.tags = torch.LongTensor(ats.get_tags())
        data_object.sid = str(p)
        data_object.descriptor = p.stem
        if p.stem in dft_adsorption_energies.keys():
            data_object.y = dft_adsorption_energies[p.stem]

            # if p.stem in dft_adsorption_energies.keys():
            #     ats.info.update({"dft_energy": dft_adsorption_energies[p.stem]})
            #     db.write(
            #         ats,
            #         data={"info": ats.info},
            #         y=dft_adsorption_energies[p.stem],
            #         name=p.stem,
            #     )

            txn = db.begin(write=True)
            txn.put(
                f"{i}".encode("ascii"),
                pickle.dumps(data_object, protocol=-1),
            )
            txn.commit()
            i += 1
        # else:
        #     conn.write(ats, data={"info": ats.info}, name=p.stem)


dataset = LmdbDataset({"src": str(lmdb_dir)})


for i in range(len(dataset)):
    print(dataset[i])
