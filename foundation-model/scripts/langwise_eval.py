import os
import sys
import argparse
import torch
import pandas as pd
import sacrebleu
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from text_protection import protect_text, restore_text

# ======================================================
# IMMEDIATE STDOUT FLUSH (IMPORTANT)
# ======================================================
sys.stdout.reconfigure(line_buffering=True)

# ======================================================
# PATHS
# ======================================================
BASE_DIR = "/home/jovyan/final-climb-shashwat-do-not-delete/foundation-model"
DATA_DIR = "/home/jovyan/final-climb-shashwat-do-not-delete/datasets/foundation-model"

# ======================================================
# MODEL MAP
# ======================================================
MODEL_MAP = {
    "nllb-200-600m": "facebook/nllb-200-600M",
    "nllb-200-1.3B": "facebook/nllb-200-1.3B",
    "nllb-200-3.3B": "facebook/nllb-200-3.3B",
}

TRIBAL_PREFIXES = ["bhi", "mun", "gon", "san", "gar", "kui"]

# ======================================================
# ARGS
# ======================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model_variant", choices=list(MODEL_MAP.keys()), required=True)
parser.add_argument("--use_lora", action="store_true")
parser.add_argument("--num_samples", type=int, default=200)
args = parser.parse_args()

if args.use_lora and args.model_variant != "nllb-200-3.3B":
    raise ValueError("--use_lora only supported for nllb-200-3.3B")

print("🚀 Starting langwise_eval.py", flush=True)

# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}", flush=True)
if device.type == "cuda":
    print(f"🧠 GPU: {torch.cuda.get_device_name(0)}", flush=True)

# ======================================================
# RESOLVE MODEL DIR
# ======================================================
variant_name = args.model_variant.replace(".", "_")
suffix = "lora" if args.use_lora else "full"

MODEL_DIR = os.path.join(BASE_DIR, "outputs", f"{variant_name}_{suffix}")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "eval")
OUT_PATH = os.path.join(OUT_DIR, f"langwise_eval_{variant_name}_{suffix}.csv")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"📦 Loading model from: {MODEL_DIR}", flush=True)

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
if args.num_samples:
    df = df.sample(n=args.num_samples, random_state=42)

print(f"📄 Loaded {len(df)} evaluation samples", flush=True)

# ======================================================
# LOAD TOKENIZER
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

# ======================================================
# LOAD MODEL
# ======================================================
if args.use_lora:
    from peft import PeftModel

    base = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_MAP[args.model_variant],
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, MODEL_DIR, local_files_only=True)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        local_files_only=True,
    )

model.to(device).eval()

# ======================================================
# HELPERS
# ======================================================
def infer_tribal(lang):
    for p in TRIBAL_PREFIXES:
        if lang.startswith(p):
            return p
    return "hub"

def infer_direction(src, tgt):
    return "hub_to_tribal" if infer_tribal(tgt) != "hub" else "tribal_to_hub"

def beam_size_for_direction(direction):
    return 5 if direction == "hub_to_tribal" else 3

# ======================================================
# EVALUATION LOOP
# ======================================================
print("🔍 Starting sentence-level evaluation loop", flush=True)

rows = []
agg = defaultdict(lambda: {"refs": [], "hyps": []})

pbar = tqdm(
    df.itertuples(index=False),
    total=len(df),
    desc="🧪 Langwise sentence eval",
    dynamic_ncols=True,
)

for row in pbar:
    src, mapping = protect_text(row.source_sentence)

    tokenizer.src_lang = row.source_lang
    tokenizer.tgt_lang = row.target_lang

    direction = infer_direction(row.source_lang, row.target_lang)
    beam_size = beam_size_for_direction(direction)

    pbar.set_postfix({
        "dir": direction,
        "beam": beam_size,
        "src": row.source_lang,
        "tgt": row.target_lang,
    }, refresh=False)

    inputs = tokenizer(
        src,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(device)

    forced_bos = tokenizer.convert_tokens_to_ids(row.target_lang)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=128,
            num_beams=beam_size,
            early_stopping=True,
        )

    pred = restore_text(
        tokenizer.decode(outputs[0], skip_special_tokens=True),
        mapping,
    )

    ref = row.target_sentence

    bleu = sacrebleu.sentence_bleu(pred, [ref]).score
    chrf = sacrebleu.sentence_chrf(pred, [ref]).score

    tribal = (
        infer_tribal(row.source_lang)
        if infer_tribal(row.source_lang) != "hub"
        else infer_tribal(row.target_lang)
    )

    rows.append({
        "Tribal": tribal,
        "Direction": direction,
        "Source Language": row.source_lang,
        "Target Language": row.target_lang,
        "Beam Size": beam_size,
        "Source Sentence": row.source_sentence,
        "Target Sentence (Actual)": ref,
        "Target Sentence (Predicted)": pred,
        "BLEU (sentence)": bleu,
        "chrF++ (sentence)": chrf,
    })

    key = f"{tribal}_{direction}"
    agg[key]["refs"].append(ref)
    agg[key]["hyps"].append(pred)

# ======================================================
# SAVE SENTENCE-LEVEL RESULTS
# ======================================================
pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
print(f"✅ Sentence-level results saved → {OUT_PATH}", flush=True)

# ======================================================
# SAVE CORPUS-LEVEL SUMMARY
# ======================================================
summary_rows = []
for k, v in agg.items():
    summary_rows.append({
        "Tribal_Direction": k,
        "Corpus BLEU": sacrebleu.corpus_bleu(v["hyps"], [v["refs"]]).score,
        "Corpus chrF++": sacrebleu.corpus_chrf(v["hyps"], [v["refs"]]).score,
    })

summary_path = OUT_PATH.replace(".csv", "_summary.csv")
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

print(f"📊 Corpus-level summary saved → {summary_path}", flush=True)
print("🏁 langwise_eval.py completed successfully", flush=True)
