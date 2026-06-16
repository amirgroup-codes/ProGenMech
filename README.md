# ProGenMech: Circuit Tracing in Autoregressive Protein Language Models
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ProGenMech_Logo_Dark.svg">
    <img alt="ProGenMech Logo" src="ProGenMech_Logo_Light.svg" width="60%">
  </picture>
</p>

This is the official code repository for the paper "Circuit Tracing in Autoregressive Protein Language Models", by Darin Tsui, William Deinzer, Daniel Saeedi, and Amirali Aghazadeh, accepted into the **Mechanistic Interpretability Workshop at ICML 2026**. A link to the paper can be found [here](https://arxiv.org/abs/2606.16044). 

Additionally, one can explore protein circuits through our [web-based visualizer](https://protmech.github.io/)!

## Quick Start

The easiest way to get started with ProtoMech is through our interactive [Google Colab notebook](https://colab.research.google.com/github/amirgroup-codes/ProGenMech/blob/main/ProGenMech.ipynb). No local installation is required.

### Workflow 

1. **Models**: ProGenMech currently supports ProGen3-Small!
2. **Circuit Discovery**: Identify circuits in either generation or zero-shot.
3. **Interactive Visualization**: Generate files required for our [website](https://protmech.github.io/) and visualize circuits!

## Main Folders
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

## Previous Data

You can find the models at [https://huggingface.co/darintsui/ProGenMechModels](https://huggingface.co/darintsui/ProGenMechModels  ) and the data used in this paper at [https://huggingface.co/datasets/darintsui/ProGenMechData](https://huggingface.co/datasets/darintsui/ProGenMechData). 

For training sequences and other data not listed above, we utilize data released from ProtoMech at [https://huggingface.co/datasets/ktalreja/ProtoMechData](https://huggingface.co/datasets/ktalreja/ProtoMechData).