"""A scrip to test the ammount of time for a gnn."""
import sys
import time

from pathlib import Path

sys.path.append("src")
from search.reward.simulation_reward import StructureReward  # noqa:E402

with open("gnn_timing_info_big_batch.txt", "w") as f:
    found_batch_size = False
    batch_size = 40
    while not found_batch_size and batch_size > 0:
        try:
            sr = StructureReward(
                **{
                    "llm_function": None,
                    "model": "gemnet",
                    "traj_dir": Path("data/output/cuda_test_big_batch"),
                    "device": "cuda",
                    "steps": 150,
                    "ads_tag": 2,
                    "num_adslab_samples": 16,
                    "batch_size": batch_size,
                }
            )
            start = time.time()
            print(
                sr.create_structures_and_calculate(
                    [["Cu"], ["Pt"], ["Ru"], ["Zn"], ["Zr"]],
                    ["*CO", "*O", "*OH2"],
                    ["Cu", "Pt", "Ru", "Zn", "Zr"],
                    placement_type="heuristic",
                )
            )
            end = time.time()
            print(f"cuda time:\t{end-start}")
            f.write(f"cuda time:\t{end-start}")
            found_batch_size = True
        except Exception as err:
            print(err)
            print(f"Batch size {batch_size} did not work.")
            batch_size -= 20
    # sr = StructureReward(
    #     **{
    #         "llm_function": None,
    #         "model": "gemnet",
    #         "traj_dir": Path("data/output/cpu_test_omp"),
    #         "device": "cpu",
    #         "steps": 150,
    #         "ads_tag": 2,
    #         "num_adslab_samples": 16,
    #     }
    # )
    # start = time.time()
    # print(
    #     sr.create_structures_and_calculate(
    #         [["Cu"], ["Pt"], ["PtZn"]],
    #         ["*CO", "*O"],
    #         ["Cu", "Pt", "PtZn"],
    #         placement_type="heuristic",
    #     )
    # )
    # end = time.time()
    # print(f"cpu time:\t{end-start}")
    # f.write(f"cpu time:\t{end-start}")
