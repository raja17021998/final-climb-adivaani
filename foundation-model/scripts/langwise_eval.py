# langwise_eval.py
import os
import torch
import pandas as pd
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from text_protection import protect_text, restore_text
import evaluate
import config

# ======================================================
# DDP SETUP
# ======================================================
RANK = int(os.environ.get("RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))

if WORLD > 1 and not dist.is_initialized():
    dist.init_process_group(backend=config.DDP_BACKEND)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# LOAD HF METRICS
# ======================================================
bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

# ======================================================
# HELPERS
# ======================================================
def infer_lang(tok, table):
    for name, prefixes in table.items():
        if any(tok.startswith(p) for p in prefixes):
            return name
    return None

def infer_route(src, tgt):
    sh = infer_lang(src, config.HUB_LANGS)
    th = infer_lang(tgt, config.HUB_LANGS)
    st = infer_lang(src, config.TRIBAL_LANGS)
    tt = infer_lang(tgt, config.TRIBAL_LANGS)

    if sh and tt:
        return "hub_to_tribal"
    if st and th:
        return "tribal_to_hub"
    return "other"

def infer_tribal(src, tgt):
    st = infer_lang(src, config.TRIBAL_LANGS)
    tt = infer_lang(tgt, config.TRIBAL_LANGS)
    return st or tt or "hub"

# ======================================================
# PATHS
# ======================================================
variant = config.MODEL_VARIANT.replace(".", "_")
suffix = "lora" if config.USE_LORA else "full"

MODEL_DIR = os.path.join(
    config.OUTPUT_DIR,
    f"{variant}_{suffix}",
    "final_model_weight"
)

df = pd.read_csv(os.path.join(config.DATA_DIR, "val.csv"))

if config.MODE == "debug":
    df = df.sample(
        n=min(len(df), config.TEST_DEBUG_ROWS),
        random_state=config.SEED,
    )

# DDP split
df = df.iloc[RANK::WORLD]

# ======================================================
# LOAD MODEL
# ======================================================
tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_DIR,
    dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    local_files_only=True,
).to(device).eval()

rows = []

iterator = tqdm(
    df.iterrows(),
    total=len(df),
    desc=f"Rank {RANK} Eval",
    dynamic_ncols=True,
)

# ======================================================
# EVALUATION LOOP
# ======================================================
for idx, r in iterator:

    src_protected, mapping = protect_text(r[config.SOURCE_COL])


    tok.src_lang = r[config.SOURCE_LANG_COL]
    tok.tgt_lang = r[config.TARGET_LANG_COL]


    direction = infer_route(r[config.SOURCE_LANG_COL], r[config.TARGET_LANG_COL])
    tribal = infer_tribal(r[config.SOURCE_LANG_COL], r[config.TARGET_LANG_COL])


    beam = config.BEAM_BY_ROUTE.get(direction, config.DEFAULT_BEAM)

    iterator.set_postfix(
        {"dir": direction, "beam": beam},
        refresh=False,
    )

    inp = tok(
        src_protected,
        return_tensors="pt",
        truncation=True,
        max_length=config.GEN_MAX_LEN,

    ).to(device)

    forced = tok.convert_tokens_to_ids(r[config.TARGET_LANG_COL])


    with torch.no_grad():
        out = model.generate(
            **inp,
            num_beams=beam,
            forced_bos_token_id=forced,
            max_length=config.MAX_LEN,
        )

    pred = restore_text(
        tok.decode(out[0], skip_special_tokens=True),
        mapping,
    )

    pred = str(pred).strip()
    ref = str(r[config.TARGET_COL]).strip()

    # ======================================================
    # SAFE METRIC COMPUTATION
    # ======================================================
    if not pred:
        bleu_score = 0.0
        chrf_score = 0.0
    else:
        try:
            bleu_score = bleu_metric.compute(
                predictions=[pred],
                references=[[ref]],
            )["bleu"] * 100

            chrf_score = chrf_metric.compute(
                predictions=[pred],
                references=[ref],
            )["score"]

        except Exception:
            bleu_score = 0.0
            chrf_score = 0.0

    rows.append({
        "Tribal": tribal,
        "Direction": direction,
        "Source Language": r[config.SOURCE_LANG_COL],
        "Target Language": r[config.TARGET_LANG_COL],
        "Source Sentence": r[config.SOURCE_COL],
        "Target Sentence (Actual)": ref,
        "Target Sentence (Predicted)": pred,
        "BLEU (sentence)": bleu_score,
        "chrF++ (sentence)": chrf_score,
    })

# ======================================================
# DDP GATHER
# ======================================================
if WORLD > 1:
    gathered = [None] * WORLD
    dist.all_gather_object(gathered, rows)
else:
    gathered = [rows]

# ======================================================
# SAVE
# ======================================================
if RANK == 0:
    flat = [x for g in gathered for x in g]

    model_output_dir = os.path.join(
        config.OUTPUT_DIR,
        f"{variant}_{suffix}"
    )

    eval_dir = os.path.join(model_output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    final_path = os.path.join(
        eval_dir,
        f"langwise_eval_{variant}_{suffix}.csv"
    )

    pd.DataFrame(flat).to_csv(
        final_path,
        index=False,
    )

    print("\n✅ Langwise evaluation complete (HF evaluate, safe).")
    print(f"📁 File saved at: {final_path}\n")
