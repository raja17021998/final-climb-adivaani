# config.py
import os

BASE_DIR = "/home/jovyan/final-climb-shashwat-do-not-delete"
PROJECT_DIR = os.path.join(BASE_DIR, "baseline-funs/lstm-lstm")

DATA_DIR = os.path.join(BASE_DIR, "datasets")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")

MODEL_SAVE_DIR = os.path.join(PROJECT_DIR, "model_weights")
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")

TOKENIZER_MODEL = os.path.join(BASE_DIR, "tokenization/joint_spm.model")
TOKENIZER_VOCAB = os.path.join(BASE_DIR, "tokenization/joint_spm.vocab")

CORPUS_DIR = os.path.join(BASE_DIR, "tokenization/corpora")

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
NEW_SCRIPT_LANGS = {}

SEQ2SEQ_PARAMS = {
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_layers": 2,
    "dropout": 0.3,
    "teacher_forcing_ratio": 0.5,
    "vocab_size": 32000,
}

TRAINING_PARAMS = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 20,
    "max_length": 128,
}

DIRECTION_CONFIG = {
    "bhili": {
        "english_bhili": True, "hindi_bhili": True, "marathi_bhili": True, "gujarati_bhili": True,
        "bhili_english": True, "bhili_hindi": True, "bhili_marathi": False, "bhili_gujarati": False,
    },
    "mundari": {
        "english_mundari": False, "hindi_mundari": True,
        "mundari_english": False, "mundari_hindi": True,
    },
    "gondi": {
        "english_gondi": True, "hindi_gondi": False,
        "gondi_english": True, "gondi_hindi": False,
    },
    "santali": {
        "english_santali": False, "hindi_santali": True,
        "santali_english": False, "santali_hindi": True,
    },
    "kui": {
        "english_kui": True, "hindi_kui": False,
        "kui_english": False, "kui_hindi": True,
    },
    "garo": {
        "english_garo": True, "hindi_garo": False,
        "garo_english": False, "garo_hindi": True,
    },
}

# ============================
# OPTIONAL DATA LIMIT PER DIRECTION
# (None = use full dataset)
# ============================
DIRECTION_DATA_LIMIT = {
    "bhili": {
        "english_bhili": 3000,
        "hindi_bhili": 3000,
        "marathi_bhili": 3000,
        "gujarati_bhili": 3000,
        "bhili_english": 3000,
        "bhili_hindi": 3000,
        "bhili_marathi": 3000,
        "bhili_gujarati": 3000,
    },
    "mundari": {
        "english_mundari": 3000,
        "hindi_mundari": 3000,
        "mundari_english": 1300,
        "mundari_hindi": 3000,
    },
    "gondi": {
        "english_gondi": 3000,
        "hindi_gondi": 3000,
        "gondi_english": 3000,
        "gondi_hindi": 3000,
    },
    "santali": {
        "english_santali": 3000,
        "hindi_santali": 3000,
        "santali_english": 3000,
        "santali_hindi": 3000,
    },
    "kui": {
        "english_kui": 1300,
        "hindi_kui": 3000,
        "kui_english": 3000,
        "kui_hindi": 1300,
    },
    "garo": {
        "english_garo": 3000,
        "hindi_garo": 3000,
        "garo_english": 3000,
        "garo_hindi": 3000,
    },
}


VAL_SPLIT = 0.1
DEBUG_MODE = True
DEBUG_TRAIN_ROWS = 10000
DEBUG_TEST_ROWS = 200

BEAM_WIDTH= 3
