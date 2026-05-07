#! /bin/bash

pip install -e .
pip install "megablocks[gg]==0.7.0" --no-build-isolation
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation
