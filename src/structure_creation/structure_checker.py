"""Utilities for atomistic structures."""

from pathlib import Path

import numpy as np
import pandas as pd
from ase.calculators.singlepoint import SinglePointCalculator

from ase import Atoms
from ase.io import Trajectory, write

from ocdata.utils.flag_anomaly import DetectTrajAnomaly

from tqdm import tqdm

if __name__ == "__main__":
    save_dir = Path("cu_zn_dft_structures")
    save_dir.mkdir(exist_ok=True)
    xyz_directory = Path("cu_zn_with_H/trajectories")
    print(xyz_directory)
    traj_files = list(xyz_directory.rglob("*.traj"))

    codes = []
    bad_counter = 0
    good_counter = 0
    for f in tqdm(traj_files):
        traj = Trajectory(str(f))
        anomaly_detector = DetectTrajAnomaly(
            init_atoms=traj[0],
            final_atoms=traj[-1],
            atoms_tag=traj[0].get_tags(),
        )
        fmax = np.max(np.sqrt(np.sum(traj[-1].get_forces() ** 2, axis=1)))
        code = None
        if anomaly_detector.has_surface_changed():
            code = 3
        elif 2 in traj[0].get_tags() and anomaly_detector.is_adsorbate_dissociated():
            code = 1
        elif 2 in traj[0].get_tags() and anomaly_detector.is_adsorbate_desorbed():
            code = 2
        elif 2 in traj[0].get_tags() and anomaly_detector.is_adsorbate_intercalated():
            code = 5
        elif fmax > 0.03:
            code = 6
        else:
            code = 0
        codes.append({f.stem: code})
        if (
            anomaly_detector.has_surface_changed()
            or fmax > 0.03
            or (
                2 in traj[0].get_tags()
                and any(
                    [
                        (
                            anomaly_detector.is_adsorbate_dissociated()  # adsorbate is dissociated
                        ),
                        (
                            anomaly_detector.is_adsorbate_desorbed()  # flying off the surfgace
                        ),
                        (
                            anomaly_detector.is_adsorbate_intercalated()  # interacting with bulk atom
                        ),
                    ]
                )
            )
        ):
            bad_counter += 1
        else:

            good_counter += 1
        xyz_path = save_dir / (f.parent.stem) / (f.stem + ".xyz")
        xyz_path.parent.mkdir(parents=True, exist_ok=True)
        write(str(xyz_path), traj[-1])
pd.DataFrame(codes).to_csv("convergence_error_codes.csv", index=False)

print(
    f"bad_trajectories: {bad_counter}, good_trajectories: {good_counter} with {bad_counter + good_counter} total structures"
)
