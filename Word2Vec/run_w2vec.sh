#!/bin/bash
set -e

# ============================================================
# CONFIG
# ============================================================

BASE="/home/jovyan/final-climb-shashwat-do-not-delete/Word2Vec"
PYTHON=$(which python)

echo "Using python: $PYTHON"
echo "Word2Vec dir: $BASE"

# ============================================================
# GPU DETECTION (SAFE)
# ============================================================

if command -v nvidia-smi &> /dev/null; then
    GPUS=$(nvidia-smi -L | wc -l)
else
    GPUS=0
fi

echo "Detected GPUs: $GPUS"

# ============================================================
# LANGUAGES (Tribal Only)
# ============================================================

LANGS=("Bhili" "Santali" "Mundari" "Gondi" "Kui" "Garo")

# ============================================================
# TRAIN LOOP
# ============================================================

for LANG in "${LANGS[@]}"
do
  echo ""
  echo "========================================"
  echo "🚀 Training $LANG"
  echo "========================================"

  if [ "$GPUS" -gt 1 ]; then
    torchrun --standalone \
             --nproc_per_node=$GPUS \
             "$BASE/train_w2vec.py" \
             --lang "$LANG"
  else
    $PYTHON "$BASE/train_w2vec.py" \
            --lang "$LANG"
  fi

  echo "✅ Completed $LANG"
done

echo ""
echo "🎉 ALL TRAININGS COMPLETED"
