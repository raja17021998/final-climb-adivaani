#!/bin/bash
set -e

BASE="/home/jovyan/final-climb-shashwat-do-not-delete/BERT"
PYTHON="/home/jovyan/miniconda3/envs/starkai/bin/python"

GPUS=$(nvidia-smi -L | wc -l)
echo "Detected GPUs: $GPUS"
echo "Using python: $PYTHON"

# -------------------------------
# CONFIG (easy to tweak)
# -------------------------------
EPOCHS=50
MAX_LINES=""   # set to empty "" or remove flag for full corpus

# Early Stopping Config
EARLY_STOPPING_PATIENCE=3
EARLY_STOPPING_MIN_DELTA=0.0

BATCH_SIZE=512


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
        --max_lines $MAX_LINES \
        --early_stopping_patience $EARLY_STOPPING_PATIENCE \
        --early_stopping_min_delta $EARLY_STOPPING_MIN_DELTA \
        --batch_size $BATCH_SIZE

    else
      torchrun --standalone --nproc_per_node=$GPUS \
        $BASE/train_bert.py \
        --lang $LANG \
        --epochs $EPOCHS \
        --early_stopping_patience $EARLY_STOPPING_PATIENCE \
        --early_stopping_min_delta $EARLY_STOPPING_MIN_DELTA \
        --batch_size $BATCH_SIZE

    fi
  else
    if [ -n "$MAX_LINES" ]; then
      $PYTHON $BASE/train_bert.py \
        --lang $LANG \
        --epochs $EPOCHS \
        --max_lines $MAX_LINES \
        --early_stopping_patience $EARLY_STOPPING_PATIENCE \
        --early_stopping_min_delta $EARLY_STOPPING_MIN_DELTA \
        --batch_size $BATCH_SIZE

    else
      $PYTHON $BASE/train_bert.py \
        --lang $LANG \
        --epochs $EPOCHS \
        --early_stopping_patience $EARLY_STOPPING_PATIENCE \
        --early_stopping_min_delta $EARLY_STOPPING_MIN_DELTA \
        --batch_size $BATCH_SIZE

    fi
  fi
}

run_lang Bhili
# run_lang Santali
# run_lang Mundari
# run_lang Gondi
# run_lang Kui

echo "ALL BERT TRAININGS COMPLETED"
