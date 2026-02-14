#!/bin/bash
set -e

BASE="/home/jovyan/final-climb-shashwat-do-not-delete/GLoVE"
PYTHON=$(which python)

echo "Using python: $PYTHON"
echo "Word2Vec dir: $BASE"

GPUS=$(nvidia-smi -L | wc -l)
echo "Detected GPUs: $GPUS"

run_lang () {
  LANG=$1
  echo "=============================="
  echo "Training GLoVe for $LANG"
  echo "=============================="

  if [ "$GPUS" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$GPUS \
      $BASE/train_glove.py --lang $LANG
  else
    $PYTHON $BASE/train_glove.py --lang $LANG
  fi
}

run_lang bhili
run_lang santali
run_lang mundari
run_lang gondi
run_lang kui
run_lang garo

echo "ALL GLoVE TRAININGS COMPLETED"
