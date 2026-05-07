# Generation Circuit Workflow

This folder contains scripts for generating sequence data and discovering generation circuits.

## Key files
- `01_prepare_data.py` — Prepare input data for generation experiments.
- `02_discover_circuits.py` — Run discovery over generated sequences.
- `main.sh` — Example wrapper for the generation workflow.
- `generation_utils.py` — Shared generation utilities.

## Run
- `sh generation_circuit/main.sh`
- Modify `ROOT_DIR` and `PYTHON_BIN` in `main.sh` to point to your local project and Python installation.
