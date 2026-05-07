# ProGen3

This repository contains code to access the ProGen3 family of models as well as cli interface to run these models for scoring and generation. The model weights are released under a **non-commercial** usage license. Please see the [LICENSE](#license) section below.

## Models Available

More info on ProGen3 Models can be found [here](https://profluent.bio/showcase/progen3). In the subsequent sections, you can provide a model name from following table where needed.

| Model Name                 | Model Parameters | HuggingFace Link                                                                |
| -------------------------- | ---------------- | ------------------------------------------------------------------------------- |
| Profluent-Bio/progen3-112m | 112M             | [Profluent-Bio/progen3-112m](https://huggingface.co/Profluent-Bio/progen3-112m) |
| Profluent-Bio/progen3-219m | 219M             | [Profluent-Bio/progen3-219m](https://huggingface.co/Profluent-Bio/progen3-219m) |
| Profluent-Bio/progen3-339m | 339M             | [Profluent-Bio/progen3-339m](https://huggingface.co/Profluent-Bio/progen3-339m) |
| Profluent-Bio/progen3-762m | 762M             | [Profluent-Bio/progen3-762m](https://huggingface.co/Profluent-Bio/progen3-762m) |
| Profluent-Bio/progen3-1b   | 1B               | [Profluent-Bio/progen3-1b](https://huggingface.co/Profluent-Bio/progen3-1b)     |
| Profluent-Bio/progen3-3b   | 3B               | [Profluent-Bio/progen3-3b](https://huggingface.co/Profluent-Bio/progen3-3b)     |

## Installation

### Local Usage

ProGen3 family of models require atleast one GPU device (we have tested it only on a A100/H100 with atleast 40GB of VRAM; GPUs from prior generations may not work due to lack of support for bf16 precision and flash attention kernels) be available.

1. Clone this repo and run `bash setup.sh`

### Docker Usage

We also provide a docker image `ghcr.io/profluent-ai/progen3:v0.1.0` that comes with certain libraries pre-installed to reduce installation time. This image still needs to be run on a machine with GPU device.

Within the container, follow the steps in the section [Local Usage](#local-usage).

## Interactive Usage

```python
import torch

from progen3.modeling import ProGen3ForCausalLM
from progen3.batch_preparer import ProGen3BatchPreparer
from progen3.scorer import ProGen3Scorer

model = ProGen3ForCausalLM.from_pretrained("Profluent-Bio/progen3-3b", torch_dtype=torch.bfloat16)
model = model.eval().to("cuda:0")
batch_preparer = ProGen3BatchPreparer()

# Direct Usage
sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"

inputs = batch_preparer.get_batch_kwargs([sequence], device="cuda:0", reverse=False)
outputs = model(**inputs, return_dict=True)
print(outputs.logits)

# Usage with scorer (returns averaged log likelihood of forward and reverse direction)
# Would suggest using Scoring CLI below if scoring very large number of sequences
scorer = ProGen3Scorer(model=model)
scores = scorer.score_batch(sequences=[sequence])
print(scores["log_likelihood"][0])
```

## Scoring CLI

Prepare the fasta file (described in the next section) containing sequences to be scored. Then, run the following command (modifying the arguments as necessary):

```bash
torchrun --nproc-per-node=gpu -m progen3.tools.score \
--fasta-path sequences.fasta \
--output-path scores.csv \
--model-name Profluent-Bio/progen3-3b \
--fsdp
```

See an example in directory [examples/protein_gym_zero_shot](examples/protein_gym_zero_shot) on how we use it to evaluate spearman score on protein gym assays.

```bash
bash examples/protein_gym_zero_shot/download_and_prepare.sh

# Run for individual assay
bash examples/protein_gym_zero_shot/run.sh A0A140D2T1_ZIKV_Sourisseau_2019 Profluent-Bio/progen3-3b

# Run for all assays and print aggregated spearman
bash examples/protein_gym_zero_shot/run.sh all Profluent-Bio/progen3-3b
```

### Input File Formats

- `fasta-path` : A path to fasta file

Each fasta file should be of form:

```
>seq_id_1
AASNNMETYR
>seq_id_2
MKKLPSDDEFGHKKL
```

where each sequence has a unique id and maximum length of 8190 residues. Once the scoring is complete, the output csv file will have following columns:

- `sequence_id` : The sequence id corresponding to each sequence in the input fasta file
- `log_likelihood` : The log likelihood of the sequence.
- `perplexity` : The perplexity of the sequence.

## Generation CLI

Prepare the prompt file (described in the next section). Then run the following command (modifying the arguments as necessary):

```bash
mkdir -p generation_outputs/

torchrun --nproc-per-node=gpu -m progen3.tools.generate \
  --prompt-file examples/generations/prompts.csv \
  --output-dir generation_outputs/ \
  --model-name Profluent-Bio/progen3-3b \
  --n-per-prompt 5000 \
  --fsdp \
  --temperature 0.85 \
  --top-p 0.95
```

You will get two files per prompt in the generation_outputs directory after completion. First is `{id}.gen.fasta` which contains the raw tokens generated by the model for each generated sequence (not containing the prompt). The second is `{id}.seq.fasta` where the generations are combined with prompt properly to output a valid N-to-C terminal protein sequence. Note, the `seq.fasta` files only contain a subset of `gen.fasta` file generations since some of the generations in `gen.fasta` file may be invalid.

See an example in directory [examples/generations](examples/generations) of a prompt file for generations for Deaminase given forward and reverse prefixes and for infilling a section of PETase (again both in forward and reverse direction).

### Prompt File Format

The prompt file should be a .csv file. Required columns: `id`, `sequence`, `min_new_tokens`, `max_new_tokens`. (min/max new tokens are added to the sequence length). id is the unique identifier for each prompt `sequence`.

`sequence` format: `<1/2><prompt>`

- `<1/2><prompt>` : The prompt to generate from. Assuming you want to generate for original sequence `AASNNMETYR`, the prompt should be:
  - For CLM in N-to-C (forward) direction: residues from N direction (e.g. `1AAS`)
  - For CLM in C-to-N (reverse) direction: residues from C direction (e.g. `2RYTE`)
  - For unconditional generation, you can leave the prompt part empty and only provide the direction indicator. e.g. `1` for N-to-C (forward) direction and `2` for C-to-N (reverse) direction.

### Info on GLM prompt preparation

Assume you have a original protein sequence of length `N`. Let's say you want to fill in the span from original residue from `<start pos>` to `<end pos>` (start inclusive, end exclusive, 0-indexed) but shorten/lengthen it to (on average) `M` residues. For the N-to-C (forward) direction, the prompt should be:

`1<original protein sequence in N-to-C direction>[GLM]<start pos>-<end pos>-<M>`

For the C-to-N (reverse) direction, the prompt should be:

`2<original protein sequence in C-to-N direction>[GLM]<N - end pos>-<N - start pos>-<M>`

where `<N - end pos>` and `<N - start pos>` is the start and end of the span from the opposite direction.

Note: you would want to set the `max_new_tokens` to the maximum length of the infill you want (this would be a hard constraint compared to `<M>` in the prompt format which is a soft length constraint).

Example:
Consider you want to fill in the `NNMET` part of `AASNNMETYR` with, on average, 4 residues instead of 5. So the start pos is 3 and the end pos is 8.

If you want to fill in the `NNMET` part in N-to-C direction, the prompt should be:

`1AASNNMETYR[GLM]3-8-4`

If you want to fill in the `NNMET` part in C-to-N direction, the prompt should be:

`2RYTEMNNSAA[GLM]2-7-4`

## LICENSE

The code in this repository in released under [Apache 2.0 License](LICENSE-CODE).

The model weights are released under [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt).
