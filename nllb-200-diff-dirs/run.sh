#!/bin/bash
set -e

export PYTHONPATH=$(pwd):$PYTHONPATH
PYTHON=$(which python)

if [ -z "$1" ]; then
  $PYTHON train_ddp.py
else
  torchrun --nproc_per_node=$1 train_ddp.py
fi
