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
MODE = "debug"          # "debug" | "train"
DEBUG_ROWS = 2500       # used only if MODE == "debug"

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
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
MAX_LEN = 128
SEED = 42
TEMPERATURE = 5.0

# ======================================================
# LOSS CONTROL
# ======================================================
PATIENCE = 2
LOSS_DECAY = 0.7
LOSS_FLOOR = 0.2

# ======================================================
# BEAM SEARCH (ROUTE AWARE)
# ======================================================
BEAM_BY_ROUTE = {}
DEFAULT_BEAM = 4

# ======================================================
# DDP
# ======================================================
DDP_BACKEND = "nccl"
DDP_TIMEOUT = timedelta(minutes=60)
FIND_UNUSED_PARAMETERS = False
