"""Utilities for atomistic structures."""

import json

from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from ase import Atoms
from ase.io import Trajectory, write

from ocdata.utils.flag_anomaly import DetectTrajAnomaly

from tqdm import tqdm

if __name__ == "__main__":
    save_dir = Path("cu_pd_co_to_methanol/final_structures")

    dft_structure_dir = save_dir / "dft_structures"
    initial_structure_dir = save_dir / "intial_structures"
    final_structure_dir = save_dir / "final_structures"

    save_dir.mkdir(exist_ok=True)
    traj_directory = Path("cu_pd_co_to_methanol/trajectories")
    print(traj_directory)
    traj_files = list(traj_directory.rglob("*.traj"))

    codes = {}
    code_counts = {}
    bad_counter = 0
    good_counter = 0
    for f in tqdm(traj_files):
        traj = Trajectory(str(f))
        initial_structure = traj[0]
        final_structure = traj[-1]
        anomaly_detector = DetectTrajAnomaly(
            init_atoms=initial_structure,
            final_atoms=final_structure,
            atoms_tag=initial_structure.get_tags(),
        )
        fmax = np.max(np.sqrt(np.sum(final_structure.get_forces() ** 2, axis=1)))
        code = None
        if anomaly_detector.has_surface_changed():
            code = 3
        elif (
            2 in initial_structure.get_tags()
            and anomaly_detector.is_adsorbate_dissociated()
        ):
            code = 1
        elif (
            2 in initial_structure.get_tags()
            and anomaly_detector.is_adsorbate_desorbed()
        ):
            code = 2
        elif (
            2 in initial_structure.get_tags()
            and anomaly_detector.is_adsorbate_intercalated()
        ):
            code = 5
        elif fmax > 0.03:
            code = 6
        else:
            code = 0
        if code in code_counts.keys():
            code_counts[code] += 1
        else:
            code_counts[code] = 1
        codes[f.stem] = code
        # if (
        #     anomaly_detector.has_surface_changed()
        #     or fmax > 0.03
        #     or (
        #         2 in initial_structure.get_tags()
        #         and any(
        #             [
        #                 (
        #                     anomaly_detector.is_adsorbate_dissociated()  # adsorbate is dissociated
        #                 ),
        #                 (
        #                     anomaly_detector.is_adsorbate_desorbed()  # flying off the surfgace
        #                 ),
        #                 (
        #                     anomaly_detector.is_adsorbate_intercalated()  # interacting with bulk atom
        #                 ),
        #             ]
        #         )
        #     )
        # ):
        #     bad_counter += 1
        # else:

        #     good_counter += 1
        final_path = final_structure_dir / (f.parent.stem) / (f.stem + ".xyz")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        write(str(final_path).replace("*", "+"), final_structure)

        initial_path = initial_structure_dir / (f.parent.stem) / (f.stem + ".xyz")
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        write(str(initial_path).replace("*", "+"), initial_structure)

        if code != 0 and code != 6:
            dft_path = dft_structure_dir / (f.parent.stem) / (f.stem + ".xyz")
            dft_path.parent.mkdir(parents=True, exist_ok=True)
            write(str(dft_path).replace("*", "+"), initial_structure)
        else:
            dft_path = dft_structure_dir / (f.parent.stem) / (f.stem + ".xyz")
            dft_path.parent.mkdir(parents=True, exist_ok=True)
            write(str(dft_path).replace("*", "+"), final_structure)


with open("convergence_error_codes.json", "w") as f:
    json.dump(codes, f)

# print(
#     f"bad_trajectories: {bad_counter}, good_trajectories: {good_counter} with {bad_counter + good_counter} total structures"
# )
print(code_counts)
