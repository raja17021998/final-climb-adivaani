

#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$ROOT_DIR/scripts"

GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
STAGE=$1

if [ -z "$STAGE" ]; then
  echo "❌ Please specify stage:"
  echo "   train | langwise_eval | langwise_summary_mean | langwise_summary_corpus | all"
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ======================================================
# TRAIN
# ======================================================
run_train() {
  if [ "$GPUS" -gt 1 ]; then
    torchrun --nproc_per_node="$GPUS" "$SCRIPTS_DIR/unified_nllb_train.py"
  else
    python "$SCRIPTS_DIR/unified_nllb_train.py"
  fi
}

# ======================================================
# EVAL
# ======================================================
run_langwise_eval() {
  python "$SCRIPTS_DIR/langwise_eval.py"
}

# ======================================================
# SUMMARY (MEAN)
# ======================================================
run_langwise_summary_mean() {
  python "$SCRIPTS_DIR/langwise_summary_mean.py"
}

# ======================================================
# SUMMARY (CORPUS TRUE BLEU)
# ======================================================
run_langwise_summary_corpus() {
  python "$SCRIPTS_DIR/langwise_summary_corpus.py"
}

# ======================================================
# STAGE SWITCH
# ======================================================
case "$STAGE" in
  train)
    run_train
    ;;
  langwise_eval)
    run_langwise_eval
    ;;
  langwise_summary_mean)
    run_langwise_summary_mean
    ;;
  langwise_summary_corpus)
    run_langwise_summary_corpus
    ;;
  all)
    run_train
    run_langwise_eval
    run_langwise_summary_mean
    run_langwise_summary_corpus
    ;;
  *)
    echo "❌ Unknown stage: $STAGE"
    exit 1
    ;;
esac

echo "✅ Stage '$STAGE' completed successfully."
