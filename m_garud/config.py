



from pathlib import Path
import torch 

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = Path("/home/user/Desktop/Shashwat/final-climb")
print(f"✅ Base Directory set to: {BASE_DIR}")

# =========================
# DATA PATHS
# =========================
DATASETS_DIR = BASE_DIR / "datasets"
UNIFIED_DATA_DIR = DATASETS_DIR / "unified"
SAMPLED_DATA_DIR = DATASETS_DIR / "sampled"
print(f"📂 Data Paths: Unified={UNIFIED_DATA_DIR.exists()}, Sampled={SAMPLED_DATA_DIR.exists()}")

# DEVICE ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Computing Device: {DEVICE}")

# =========================
# BERT MODEL PATHS
# =========================
BERT_BASE_DIR = BASE_DIR / "BERT-based-models"
MBERT_PATH = str(BERT_BASE_DIR / "mbert")

TEACHER_BERT_PATHS = {
    "hindi": str(BERT_BASE_DIR / "hindi-bert"),
    "marathi": str(BERT_BASE_DIR / "marathi-bert"),
    "gujarati": str(BERT_BASE_DIR / "gujarati-bert"),
    "telugu": str(BERT_BASE_DIR / "telugu-bert"),
}
print(f"🤖 Teacher Models: {list(TEACHER_BERT_PATHS.keys())} paths initialized.")

# =========================
# TRIBAL ↔ TEACHER MAPPING
# =========================
LANGUAGE_MAP = {
    "bhili": {
        "pivot": "Hindi",
        "teachers": ["Hindi", "Marathi", "Gujarati"],
        "sampled_file": "Hin_Bhi_Mar_Guj_sampled.csv",
        "columns": {
            "tribal": "Bhili",
            "teachers": ["Hindi", "Marathi", "Gujarati"],
        },
    },
    "gondi": {
        "pivot": "Hindi",
        "teachers": ["Hindi", "Marathi", "Telugu"],
        "sampled_file": "Hin_Gon_Tel_Mar_sampled.csv",
        "columns": {
            "tribal": "Gondi",
            "teachers": ["Hindi", "Marathi", "Telugu"],
        },
    },
}
print(f"🗺️ Language Map: {list(LANGUAGE_MAP.keys())} mappings loaded.")

# =========================
# LANGUAGE TOKENS
# =========================
LANGUAGE_TOKENS = {
    lang: f"<LANG_{lang.upper()}>"
    for lang in LANGUAGE_MAP.keys()
}

LANG2ID = {
    lang: idx
    for idx, lang in enumerate(sorted(LANGUAGE_MAP.keys()))
}
print(f"🏷️ Language Tokens: {LANGUAGE_TOKENS}")
print(f"🆔 Language IDs: {LANG2ID}")

# =========================
# TOKENIZER CONFIG
# =========================
TOKENIZER_DIR = BASE_DIR / "m_garud" / "tokenization" / "artifacts"
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

SPM_TRAIN_DATA_DIR = SAMPLED_DATA_DIR
SPM_MODEL_PREFIX = TOKENIZER_DIR / "spm"
SPM_VOCAB_SIZE = 32000
SPM_MODEL_TYPE = "unigram"

SPECIAL_TOKENS = [
    "<MASK>",
    "<CLS>",
    "<SEP>",
    "<LANG_BHILI>",
    "<LANG_GONDI>",
]
print(f"🔡 Tokenizer: Dir created at {TOKENIZER_DIR}. Vocab Size: {SPM_VOCAB_SIZE}")

# =========================
# TOKEN IDS / SPLITS
# =========================
PAD_TOKEN_ID = 1
MASK_IGNORE_INDEX = -100
VAL_SPLIT_RATIO = 0.1
DATA_TEMP = 5
print(f"⚙️ Config: Pad ID={PAD_TOKEN_ID}, Val Split={VAL_SPLIT_RATIO}, Data Temp={DATA_TEMP}")

# =========================
# MODEL ARCHITECTURE
# =========================
HIDDEN_DIM = 768
NUM_HEADS = 12
NUM_KV_GROUPS = 4
print(f"🏗️ Architecture: Hidden={HIDDEN_DIM}, Heads={NUM_HEADS}, KV Groups={NUM_KV_GROUPS}")

# =========================
# POSITIONAL / ATTENTION
# =========================
MAX_POSITION_EMBEDDINGS = 2048
ROPE_BASE = 10000
ATTN_MASK_FILL_VALUE = -10000.0
print(f"📏 Attention: Max Pos={MAX_POSITION_EMBEDDINGS}, RoPE Base={ROPE_BASE}")

# =========================
# TRAINING HYPERPARAMETERS
# =========================
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
ADAM_BETAS = (0.9, 0.999)
BATCH_SIZE = 16
NUM_EPOCHS = 50
PATIENCE = 6
print(f"📉 Training: LR={LEARNING_RATE}, Batch Size={BATCH_SIZE}, Epochs={NUM_EPOCHS}")

# =========================
# FUSION & CONTRASTIVE (MoCo)
# =========================
FUSION_TEMPERATURE = 1.0
CONTRASTIVE_TEMPERATURE = 0.07
MOCO_QUEUE_SIZE = 4096
MOCO_EMB_DIM = HIDDEN_DIM
MOCO_PRIMARY_TEACHER = "hindi"
MOCO_WARMUP_STEPS = 0
print(f"⚡ Fusion/Contrastive: MoCo Queue={MOCO_QUEUE_SIZE}, Primary={MOCO_PRIMARY_TEACHER}")

# =========================
# SAMPLING & SPAN MLM
# =========================
DEFAULT_SAMPLE_SIZE = 1000
RANDOM_SEED = 42

SPAN_MASK_PROB = 0.15
SPAN_MIN_LEN = 2
SPAN_MAX_LEN = 5
SPAN_REPLACE_PROB = 0.8
SPAN_RANDOM_PROB = 0.1
SPAN_KEEP_PROB = 0.1
MIN_SPAN_COUNT = 1
SPAN_IGNORE_INDEX = -100
print(f"🎭 Span MLM: Mask Prob={SPAN_MASK_PROB}, Span Range=[{SPAN_MIN_LEN}, {SPAN_MAX_LEN}]")

# =========================
# EVALUATION & LOGGING
# =========================
EVAL_BATCH_SIZE = 8
RETRIEVAL_K = [1, 5, 10]
EMBED_POOLING = "mean"

LOG_DIR = BASE_DIR / "m_garud" / "logs"
LOG_DIR.mkdir(exist_ok=True)
print(f"📊 Eval/Logs: K={RETRIEVAL_K}, Pooling={EMBED_POOLING}, Log Dir Created.")

print("\n--- ✅ ALL CONFIGURATIONS LOADED SUCCESSFULLY ---")
