# Example for running reaction pathway claculations for methanol

- run `create_runs_methanol.py`` to create a list of methanol inputs to reaction_reward.py (methanol_pathways.pkl)
- then in the root directory, run `src/search/reward/reaction_reward.py --config reaction_configs/methanol.yaml`
- `reaction_configs/methanol.yaml` contains the `yaml` configuration file for `traj_dir` and the file name containing reaction pathway input.

- `create_runs_ethanol.py` is also provided in the root directory.