#!/bin/bash
set -e

BASE="/home/shashwat1/final-climb-shashwat-do-not-delete/BERT"
PYTHON="/home/shashwat1/miniconda3/envs/stark-latest/bin/python"

GPUS=$(nvidia-smi -L | wc -l)
echo "Detected GPUs: $GPUS"
echo "Using python: $PYTHON"

# -------------------------------
# CONFIG (easy to tweak)
# -------------------------------
EPOCHS=10
MAX_LINES=7500   # set to empty "" or remove flag for full corpus

run_lang () {
  LANG=$1
  echo "=============================="
  echo "Training BERT for $LANG"
  echo "=============================="

  if [ "$GPUS" -gt 1 ]; then
    if [ -n "$MAX_LINES" ]; then
      torchrun --standalone --nproc_per_node=$GPUS \
        $BASE/train_bert.py \
        --lang $LANG \
        --epochs $EPOCHS \
        --max_lines $MAX_LINES
    else
      torchrun --standalone --nproc_per_node=$GPUS \
        $BASE/train_bert.py \
        --lang $LANG \
        --epochs $EPOCHS
    fi
  else
    if [ -n "$MAX_LINES" ]; then
      $PYTHON $BASE/train_bert.py \
        --lang $LANG \
        --epochs $EPOCHS \
        --max_lines $MAX_LINES
    else
      $PYTHON $BASE/train_bert.py \
        --lang $LANG \
        --epochs $EPOCHS
    fi
  fi
}

run_lang Bhili
run_lang Santali
# run_lang Mundari
# run_lang Gondi
# run_lang Kui

echo "ALL BERT TRAININGS COMPLETED"
