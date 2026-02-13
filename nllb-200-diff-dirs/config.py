# config.py
import os

BASE_DIR = "/home/jovyan/final-climb-shashwat-do-not-delete"
PROJECT_DIR = "/home/jovyan/final-climb-shashwat-do-not-delete/nllb-200-diff-dirs"

DATA_DIR = os.path.join(BASE_DIR, "datasets")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")

MODEL_SAVE_DIR = os.path.join(PROJECT_DIR, "model_weights")
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")

TRIBAL_LANGS = ["bhili", "mundari", "gondi", "santali", "kui", "garo"]
HUB_LANGS_COMMON = ["hindi", "english"]
HUB_LANGS_BHILI = ["hindi", "english", "marathi", "gujarati"]

LANGUAGE_COLUMN_MAP = {
    "bhili": {"bhili": "Bhili", "hindi": "Hindi", "english": "English", "marathi": "Marathi", "gujarati": "Gujarati"},
    "mundari": {"mundari": "Mundari", "hindi": "Hindi", "english": "English"},
    "gondi": {"gondi": "Gondi", "hindi": "Hindi", "english": "English"},
    "santali": {"santali": "Santali", "hindi": "Hindi", "english": "English"},
    "kui": {"kui": "Kui", "hindi": "Hindi", "english": "English"},
    "garo": {"garo": "Garo", "hindi": "Hindi", "english": "English"},
}

LANGUAGE_CODE_MAP = {
    "english": "eng_Latn",
    "hindi": "hin_Deva",
    "marathi": "mar_Deva",
    "gujarati": "guj_Gujr",
    "bhili": "bhi_Deva",
    "mundari": "mun_Deva",
    "gondi": "gon_Deva",
    "santali": "sat_Olck",
    "kui": "kui_Orya",
    "garo": "grt_Latn",
}

SCRIPT_MAP = {
    "Deva": ["hin_Deva", "mar_Deva"],
    "Latn": ["eng_Latn"],
    "Gujr": ["guj_Gujr"],
    "Olck": ["sat_Olck"],
    "Orya": ["kui_Orya"],
}

LANG_TOKEN_OVERRIDES = {}

NEW_SCRIPT_LANGS = {
    # Example:
    # "xyz_NewScript": {"script": "NewScript", "init_from": ["hin_Deva", "eng_Latn"]}
}

MODEL_CHOICE = {
    "bhili": "facebook/nllb-200-distilled-600M",
    "mundari": "facebook/nllb-200-distilled-600M",
    "gondi": "facebook/nllb-200-distilled-600M",
    "santali": "facebook/nllb-200-distilled-600M",
    "kui": "facebook/nllb-200-distilled-600M",
    "garo": "facebook/nllb-200-distilled-600M",
}

USE_LORA = {
    "bhili": False,
    "mundari": False,
    "gondi": False,
    "santali": False,
    "kui": False,
    "garo": False,
}

HYPERPARAMS = {
    "bhili": {"batch_size": 8, "learning_rate": 5e-5, "num_epochs": 3, "warmup_steps": 500, "max_length": 128, "beam_width": 3},
    "mundari": {"batch_size": 8, "learning_rate": 5e-5, "num_epochs": 3, "warmup_steps": 500, "max_length": 128, "beam_width": 3},
    "gondi": {"batch_size": 8, "learning_rate": 5e-5, "num_epochs": 3, "warmup_steps": 500, "max_length": 128, "beam_width": 3},
    "santali": {"batch_size": 8, "learning_rate": 5e-5, "num_epochs": 3, "warmup_steps": 500, "max_length": 128, "beam_width": 3},
    "kui": {"batch_size": 8, "learning_rate": 5e-5, "num_epochs": 3, "warmup_steps": 500, "max_length": 128, "beam_width": 3},
    "garo": {"batch_size": 8, "learning_rate": 5e-5, "num_epochs": 3, "warmup_steps": 500, "max_length": 128, "beam_width": 3},
}

# ============================
# TRAINING DIRECTIONS CONTROL
# ============================
DIRECTION_CONFIG = {
    "bhili": {
        "english_bhili": True,
        "hindi_bhili": True,
        "marathi_bhili": True,
        "gujarati_bhili": True,
        "bhili_english": True,
        "bhili_hindi": True,
        "bhili_marathi": False,
        "bhili_gujarati": False,
    },
    "mundari": {
        "english_mundari": False,
        "hindi_mundari": True,
        "mundari_english": False,
        "mundari_hindi": True,
    },
    "gondi": {
        "english_gondi": True,
        "hindi_gondi": False,
        "gondi_english": True,
        "gondi_hindi": False,
    },
    "santali": {
        "english_santali": False,
        "hindi_santali": True,
        "santali_english": False,
        "santali_hindi": True,
    },
    "kui": {
        "english_kui": True,
        "hindi_kui": False,
        "kui_english": False,
        "kui_hindi": True,
    },
    "garo": {
        "english_garo": True,
        "hindi_garo": False,
        "garo_english": False,
        "garo_hindi": True,
    },
}

# ============================
# OPTIONAL DATA LIMIT PER DIRECTION
# (None = use full dataset)
# ============================
DIRECTION_DATA_LIMIT = {
    "bhili": {
        "english_bhili": 1000,
        "hindi_bhili": 1000,
        "marathi_bhili": 1000,
        "gujarati_bhili": 1000,
        "bhili_english": 1000,
        "bhili_hindi": 1000,
        "bhili_marathi": 1000,
        "bhili_gujarati": 1000,
    },
    "mundari": {
        "english_mundari": 1000,
        "hindi_mundari": 1000,
        "mundari_english": 1000,
        "mundari_hindi": 1000,
    },
    "gondi": {
        "english_gondi": 1000,
        "hindi_gondi": 1000,
        "gondi_english": 1000,
        "gondi_hindi": 1000,
    },
    "santali": {
        "english_santali": 1000,
        "hindi_santali": 1000,
        "santali_english": 1000,
        "santali_hindi": 1000,
    },
    "kui": {
        "english_kui": 1000,
        "hindi_kui": 1000,
        "kui_english": 1000,
        "kui_hindi": 1000,
    },
    "garo": {
        "english_garo": 1000,
        "hindi_garo": 1000,
        "garo_english": 1000,
        "garo_hindi": 1000,
    },
}

VAL_SPLIT = 0.1
DEBUG_MODE = True
DEBUG_TRAIN_ROWS = 1000
DEBUG_TEST_ROWS = 200
