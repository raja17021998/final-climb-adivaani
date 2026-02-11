#!/bin/bash
set -e

BASE="/home/shashwat1/final-climb-shashwat-do-not-delete/GLoVE"
PYTHON="/home/shashwat1/miniconda3/envs/stark-latest/bin/python"

GPUS=$(nvidia-smi -L | wc -l)
echo "Detected GPUs: $GPUS"
echo "Using python: $PYTHON"

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

# ----------------------------
# Languages
# ----------------------------
run_lang Bhili
run_lang Santali
run_lang Garo
run_lang Mundari
run_lang Gondi
run_lang Kui

echo "ALL GLoVe TRAININGS COMPLETED"
