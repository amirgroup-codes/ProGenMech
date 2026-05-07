# ProGenMech

This repository contains tools for discovering and analyzing neural network circuits in protein sequence models.

## Main folders
- `circuit_utils/` — Core CLT/PLT circuit utility code.
- `function_circuit/` — Function circuit discovery and circuit JSON outputs.
- `generation_circuit/` — Generation tasks and discovery workflows.
- `training/` — Training code for the CLT model.
- `training_transcoder/` — Training code for the PLT/transcoder model.
- `visualization/` — Analysis and visualization scripts for circuit outputs.
- `steering/` — Steering and replacement model experiments.
- `data/` — Input DMS and sequence data.
- `external/` — Vendored external ProGen3 model source.
- `models/` — Pretrained and checkpointed model directories.

## Usage
- Run discovery: `sh circuit_discovery.sh`
- Run generation analysis: `sh generation_circuit/main.sh`
- Run steering: `sh steering/main_steering.sh`
- Use visualization scripts from `visualization/`, e.g. `python visualization/get_edge_weights.py`