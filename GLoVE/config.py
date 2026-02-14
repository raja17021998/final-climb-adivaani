# ============================================================
# Central Config (Shared with BERT-style tokenizer pipeline)
# ============================================================

from pathlib import Path

BASE_DIR = Path("/home/jovyan/final-climb-shashwat-do-not-delete")

# Tokenizer
TOKENIZER_PATH = BASE_DIR / "tokenization" / "joint_spm.model"

# Corpora directory
CORPORA_DIR = BASE_DIR / "tokenization" / "corpora"

# Output
SAVE_ROOT = BASE_DIR / "GLoVE"

# Languages
LANGUAGES = ["bhili", "santali", "mundari", "gondi", "kui", "garo"]

# Hyperparameters
EMBED_DIM   = 300
WINDOW_SIZE = 8
X_MAX       = 100
ALPHA       = 0.75
BATCH_SIZE  = 131072
EPOCHS      = 10
PATIENCE    = 5
LR          = 0.05

# Special tokens to ignore in co-occurrence
IGNORE_TOKEN_IDS = {0, 1, 2, 3, 4}  # pad, bos, eos, unk, <mask>
