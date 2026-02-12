# ============================================================
# WORD2VEC CONFIG (Fully Auto SPD Driven)
# ============================================================

from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DATA_DIR = Path(
    "/home/jovyan/final-climb-shashwat-do-not-delete/datasets"
)

SAVE_ROOT = Path(
    "/home/jovyan/final-climb-shashwat-do-not-delete/Word2Vec"
)

# ============================================================
# GLOBAL FIXED PARAMETERS
# ============================================================

EMBED_DIM = 150     # SAME for all languages
BATCH_SIZE = 8192
LR = 1e-3
PATIENCE = 3

# ============================================================
# TRIBAL LANGUAGE CONFIG
# ============================================================

TRIBAL_LANG_CONFIG = {
    "Bhili": {
        "file": "Bhi_Hin_Mar_Guj_Eng.csv",
        "tribal_col": 1,   # adjust if needed
    },
    "Santali": {
        "file": "San_Hin_Eng.csv",
        "tribal_col": 1,
    },
    "Mundari": {
        "file": "Mun_Hin_Eng.csv",
        "tribal_col": 1,
    },
    "Gondi": {
        "file": "Gon_Hin_Eng.csv",
        "tribal_col": 1,
    },
    "Kui": {
        "file": "Kui_Hin_Eng.csv",
        "tribal_col": 1,
    },
    "Garo": {
        "file": "Garo_Hin_Eng.csv",
        "tribal_col": 1,
    },
}

# ============================================================
# SPD AUTO-DETECTION
# ============================================================

def compute_spd(lang: str) -> int:
    """
    Compute Sentences Per Direction dynamically
    based on unique tribal sentences.
    """
    cfg = TRIBAL_LANG_CONFIG[lang]
    path = BASE_DATA_DIR / cfg["file"]

    df = pd.read_csv(path)
    tribal_sentences = df.iloc[:, cfg["tribal_col"]].astype(str)

    return tribal_sentences.nunique()


# ============================================================
# SPD → PARAM SCALING
# ============================================================

def get_scale_params(spd: int):

    if spd >= 150_000:
        return {
            "WINDOW_SIZE": 5,
            "NEG_SAMPLES": 10,
            "MIN_COUNT": 5,
            "EPOCHS": 35,
            "SUBSAMPLE_T": 1e-5,
        }

    elif spd >= 50_000:
        return {
            "WINDOW_SIZE": 5,
            "NEG_SAMPLES": 8,
            "MIN_COUNT": 3,
            "EPOCHS": 28,
            "SUBSAMPLE_T": 1e-5,
        }

    elif spd >= 20_000:
        return {
            "WINDOW_SIZE": 4,
            "NEG_SAMPLES": 5,
            "MIN_COUNT": 2,
            "EPOCHS": 18,
            "SUBSAMPLE_T": 1e-4,
        }

    else:
        return {
            "WINDOW_SIZE": 4,
            "NEG_SAMPLES": 5,
            "MIN_COUNT": 1,
            "EPOCHS": 15,
            "SUBSAMPLE_T": 1e-4,
        }
