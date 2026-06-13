# ProGenMech: Circuit Tracing in Autoregressive Protein Language Models
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ProGenMech_Logo_Dark.svg">
    <img alt="ProGenMech Logo" src="ProtoMech_Logo_Light.svg" width="60%">
  </picture>
</p>

This is the official code repository for the paper "Circuit Tracing in Autoregressive Protein Language Models", by Darin Tsui, William Deinzer, Daniel Saeedi, and Amirali Aghazadeh, accepted into the **Mechanistic Interpretability Workshop at ICML 2026**. A link to the paper can be found [here](https://arxiv.org/abs/2602.12026). 

Additionally, one can explore protein circuits through our [web-based visualizer](https://protmech.github.io/)!

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