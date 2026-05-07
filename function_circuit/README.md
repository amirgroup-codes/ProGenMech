# Function Circuit Discovery

This folder implements function circuit discovery and scoring for CLT/PLT models.

## Key files
- `prepare_data.py` — Prepares DMS data for discovery workflows.
- `discover_circuits.py` — Main function circuit discovery pipeline.
- `clt_plt_scorer.py` — Scoring utilities for CLT and PLT circuits.
- `function_utils.py` — Shared helper functions for task generation and metrics.

## Run
- `python function_circuit/discover_circuits.py`
- Use the existing `function_circuit/circuits/` folder as the input/output circuit store.
