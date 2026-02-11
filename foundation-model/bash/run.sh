#!/bin/bash
set -e

# =========================================================
# RESOLVE PROJECT ROOT (path-safe)
# =========================================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SCRIPTS_DIR="$ROOT_DIR/scripts"
OUTPUTS_DIR="$ROOT_DIR/outputs"

# =========================================================
# CONFIG
# =========================================================
MODE="debug"            # debug | train
NUM_ROWS=1000           # only used in debug
EPOCHS=20
BATCH_SIZE=128
LR=1e-5
SEED=42

MODEL_VARIANT="nllb-200-600m"   # nllb-200-600m | nllb-200-1.3B | nllb-200-3.3B
USE_LORA=""                     # ONLY use --use_lora for 3.3B

NUM_EVAL_SAMPLES=200

GPUS=$(nvidia-smi -L | wc -l)
STAGE=$1                        # train | langwise_eval | langwise_summary | all

if [ -z "$STAGE" ]; then
  echo "❌ Please specify stage: train | langwise_eval | langwise_summary | all"
  exit 1
fi

# =========================================================
# DERIVED NAMES
# =========================================================
VARIANT_NAME="${MODEL_VARIANT//./_}"
SUFFIX="full"
if [ -n "$USE_LORA" ]; then
  SUFFIX="lora"
fi

MODEL_DIR="$OUTPUTS_DIR/${VARIANT_NAME}_${SUFFIX}"
EVAL_DIR="$MODEL_DIR/eval"

mkdir -p "$EVAL_DIR"

echo "📦 Model directory: $MODEL_DIR"

# =========================================================
# COMMON ARGS
# =========================================================
TRAIN_ARGS=(
  --mode "$MODE"
  --debug_rows "$NUM_ROWS"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --seed "$SEED"
  --model_variant "$MODEL_VARIANT"
  $USE_LORA
)

EVAL_ARGS=(
  --model_variant "$MODEL_VARIANT"
  $USE_LORA
  --num_samples "$NUM_EVAL_SAMPLES"
)

# =========================================================
# CLEAN GPU STATE (OWN PROCESSES ONLY)
# =========================================================
clean_gpu() {
  echo "🧹 Cleaning GPU state (user: $USER)..."
  pkill -u "$USER" -f torchrun || true
  pkill -u "$USER" -f unified_nllb_train.py || true
  pkill -u "$USER" -f langwise_eval.py || true
  sleep 3
  nvidia-smi
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

# =========================================================
# TRAIN
# =========================================================
run_train() {
  clean_gpu
  if [ "$GPUS" -gt 1 ]; then
    echo "🚀 Launching DDP training with $GPUS GPUs"
    torchrun --nproc_per_node="$GPUS" \
      "$SCRIPTS_DIR/unified_nllb_train.py" "${TRAIN_ARGS[@]}"
  else
    echo "🚀 Launching single-GPU training"
    python "$SCRIPTS_DIR/unified_nllb_train.py" "${TRAIN_ARGS[@]}"
  fi
}

# =========================================================
# LANGWISE EVAL (SENTENCE LEVEL)
# =========================================================
run_langwise_eval() {
  echo "📊 Running sentence-level language-wise evaluation"
  python "$SCRIPTS_DIR/langwise_eval.py" "${EVAL_ARGS[@]}"
}

# =========================================================
# LANGWISE SUMMARY (MINISTRY TABLE)
# =========================================================
run_langwise_summary() {
  INPUT_CSV="$EVAL_DIR/langwise_eval_${VARIANT_NAME}_${SUFFIX}.csv"
  OUTPUT_CSV="$EVAL_DIR/language_wise_summary.csv"

  echo "📈 Building language-wise summary table"
  python "$SCRIPTS_DIR/langwise_summary.py" \
    --input_csv "$INPUT_CSV" \
    --output_csv "$OUTPUT_CSV"
}

# =========================================================
# DISPATCH
# =========================================================
case "$STAGE" in
  train)
    run_train
    ;;
  langwise_eval)
    run_langwise_eval
    ;;
  langwise_summary)
    run_langwise_summary
    ;;
  all)
    run_train
    run_langwise_eval
    run_langwise_summary
    ;;
  *)
    echo "❌ Unknown stage: $STAGE"
    echo "Valid options: train | langwise_eval | langwise_summary | all"
    exit 1
    ;;
esac

echo "✅ Stage '$STAGE' completed successfully."
