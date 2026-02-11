#!/bin/bash
set -e

BASE="/home/kshhorizon/data/final-climb-shashwat-do-not-delete/Word2Vec"
PYTHON=$(which python)

GPUS=$(nvidia-smi -L | wc -l)
echo "Detected GPUs: $GPUS"
echo "Using python: $PYTHON"

LANGS=("Bhili" "Santali" "Mundari" "Gondi" "Kui")

for LANG in "${LANGS[@]}"
do
  echo "=============================="
  echo "Training $LANG"
  echo "=============================="

  if [ "$GPUS" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$GPUS \
      $PYTHON $BASE/train_w2vec.py --lang $LANG
  else
    $PYTHON $BASE/train_w2vec.py --lang $LANG
  fi

  wait
done

echo "ALL TRAININGS COMPLETED"
