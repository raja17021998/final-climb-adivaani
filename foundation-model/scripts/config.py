
# config.py

import os
from datetime import timedelta

# ======================================================
# BASE PATHS
# ======================================================
BASE_DIR = "/home/jovyan/final-climb-shashwat-do-not-delete/foundation-model"
DATA_DIR = "/home/jovyan/final-climb-shashwat-do-not-delete/datasets/foundation-model"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

LOG_DIR = "logs"
LOSS_DIR = "loss"

# ======================================================
# RUN MODE
# ======================================================
MODE = "debug"  # "debug" | "train"

DEBUG_ROWS = 500
TEST_DEBUG_ROWS = int(0.1 * DEBUG_ROWS)

# ======================================================
# DATA COLUMNS (FULLY CONFIG DRIVEN)
# ======================================================
SOURCE_COL = "source_sentence"
TARGET_COL = "target_sentence"
SOURCE_LANG_COL = "source_lang"
TARGET_LANG_COL = "target_lang"

# ======================================================
# LANGUAGE TOPOLOGY
# ======================================================
TRIBAL_LANGS = {
    "bhili": ["bhi"],
    "mundari": ["mun"],
    "gondi": ["gon"],
    "santali": ["san"],
    "garo": ["gar"],
    "kuii": ["kui"],
}

HUB_LANGS = {
    "hindi": ["hin"],
    "english": ["eng"],
    "marathi": ["mar"],
    "gujarati": ["guj"],
}

# ======================================================
# LANGUAGE TOKEN INITIALIZATION MAP
# new_token : initialization_token
# ======================================================
NEW_LANGUAGE_TOKEN_MAP = {
    "bhi_Deva": "mar_Deva",
    "mun_Deva": "hin_Deva",
    "gon_Deva": "hin_Deva",
    "kui_Orya": "ory_Orya",
    "gar_Latn": "eng_Latn",
    "san_Olck": "sat_Olck",  # Santali Ol Chiki initialized from Odia
}

FREEZE_NEW_LANG_TOKENS = True   # or False if you want them trainable
# ======================================================
# AUTO LANGUAGE DERIVATIONS
# ======================================================
def build_language_sets():
    all_tribal = set(TRIBAL_LANGS.keys())
    all_hub = set(HUB_LANGS.keys())
    return all_tribal, all_hub

ALL_TRIBAL, ALL_HUB = build_language_sets()


def build_valid_buckets():
    """
    Auto-generate all valid translation buckets
    without hardcoding language names anywhere else.
    """

    buckets = []

    for tribal in TRIBAL_LANGS:
        for hub in HUB_LANGS:
            buckets.append(f"{tribal}_{hub}_to_{tribal}")
            buckets.append(f"{tribal}_{tribal}_to_{hub}")

    return buckets


VALID_BUCKETS = build_valid_buckets()

# ======================================================
# DATA SPLIT
# ======================================================
TRAIN_RATIO = 0.85
VAL_RATIO = 0.05
TEST_RATIO = 0.10

# ======================================================
# MODEL
# ======================================================
MODEL_MAP = {
    "nllb-200-600m": "facebook/nllb-200-distilled-600M",
    "nllb-200-1.3B": "facebook/nllb-200-1.3B",
    "nllb-200-3.3B": "facebook/nllb-200-3.3B",
}

MODEL_VARIANT = "nllb-200-600m"
USE_LORA = False

# ======================================================
# TRAINING
# ======================================================
EPOCHS = 5
BATCH_SIZE = 80
LEARNING_RATE = 1e-5
MAX_LEN = 128
SEED = 42
TEMPERATURE = 1.67

NUM_WORKERS = 4
PIN_MEMORY = True

# ======================================================
# LOSS CONTROL
# ======================================================
PATIENCE = 2
LOSS_DECAY = 0.9
LOSS_FLOOR = 0.5

# ======================================================
# GENERATION (INFERENCE CONTROL)
# ======================================================
BEAM_BY_ROUTE = {
    # Example:
    "hub_to_tribal": 4,
    "tribal_to_hub": 2,
}

DEFAULT_BEAM = 4

GEN_MAX_LEN = MAX_LEN
GEN_NUM_BEAMS_DEFAULT = DEFAULT_BEAM

# ======================================================
# DDP
# ======================================================
DDP_BACKEND = "nccl"
DDP_TIMEOUT = timedelta(minutes=60)
FIND_UNUSED_PARAMETERS = False


ENABLE_LANG_TOKEN_EXTENSION= True 
# ======================================================
# SANITY VALIDATION (AUTO CHECK)
# ======================================================
def validate_config():
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
        "Train/Val/Test ratios must sum to 1."

    assert MODEL_VARIANT in MODEL_MAP, \
        "MODEL_VARIANT must exist inside MODEL_MAP."

    assert len(TRIBAL_LANGS) > 0, \
        "At least one tribal language required."

    assert len(HUB_LANGS) > 0, \
        "At least one hub language required."
def __main__():
    validate_config()
