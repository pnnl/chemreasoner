"""Run inference on a given lmdb dataset."""

import argparse
import sys

from pathlib import Path

from ocpmodels.datasets.lmdb_dataset import LmdbDataset

import torch
from torch_geometric.data import Batch
from torch_geometric.loader.data_list_loader import DataListLoader

sys.path.append("src")
from nnp.oc import OCAdsorptionCalculator

ads_calc = OCAdsorptionCalculator(
    **{
        "model": "gemnet-t",
        "traj_dir": Path("irrelevant"),
        "batch_size": 40,
        "device": "cpu",
        "ads_tag": 2,
        "fmax": 0.05,
        "steps": 300,
    }
)

torch_calc = ads_calc.get_torch_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb-dir", type=str)
    parser.add_argument("--batch-size", type=int, default=40)

    args = parser.parse_args()

    dataset = LmdbDataset({"src": args.lmdb_dir})
    loader = DataListLoader(dataset, batch_size=args.batch_size, shuffle=False)
    data_list = []
    for data_list in loader:
        batch = Batch.from_data_list(data_list)
        print(batch)
        print(torch_calc.model)
        print(
            torch_calc.predict(
                batch,
                per_image=False,
            )
        )
