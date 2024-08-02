"""Utilities for atomistic structures."""

from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from ase import Atoms
from ase.io import Trajectory

from ocdata.utils.flag_anomaly import DetectTrajAnomaly

from tqdm import tqdm


if __name__ == "__main__":
    xyz_directory = Path("../cu_zn_test/trajectories")
    print(xyz_directory)
    traj_files = list(xyz_directory.rglob("*.traj"))

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
            ats = traj[0].copy()
            res = traj[0].calc.results
            res["energy"] = np.nan
            ats.calc = SinglePointCalculator(traj[0], **res)
            print(ats.get_potential_energy())
        else:
            good_counter += 1 if 2 in traj[0].get_tags() else 0

print(
    f"bad_trajectories: {bad_counter}, good_trajectories: {good_counter} with {bad_counter + good_counter} total structures"
)
