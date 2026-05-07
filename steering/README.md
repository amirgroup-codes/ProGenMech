# Steering Experiments

This folder contains the steering and replacement model pipelines.

## Key files
- `run_steering.py` — Main steering driver.
- `main_steering.sh` — Example pipeline wrapper for steering experiments.
- `main_probe_steering.sh` — Example probe steering workflow.
- `steering_utils.py` — Utility helpers for loading DMS data and steering models.
- `local_replacement_models.py` — Local replacement model utilities.
- `full_replacement_models.py` — Full replacement model utilities.

## Run
- `sh steering/main_steering.sh` or `sh steering/main_probe_steering.sh`
- Update `ROOT_DIR` and `PYTHON_BIN` placeholders before running.
