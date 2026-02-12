# inference_csv.py
import os
import pandas as pd
from tqdm import tqdm
from inference import translate
import config


# ======================================================
# LANGUAGE HELPERS
# ======================================================
def _infer_lang(token, table):
    for name, prefixes in table.items():
        if any(token.startswith(p) for p in prefixes):
            return name
    return None


def _infer_route(src_lang, tgt_lang):
    sh = _infer_lang(src_lang, config.HUB_LANGS)
    th = _infer_lang(tgt_lang, config.HUB_LANGS)
    st = _infer_lang(src_lang, config.TRIBAL_LANGS)
    tt = _infer_lang(tgt_lang, config.TRIBAL_LANGS)

    if sh and tt:
        return "hub_to_tribal"
    if st and th:
        return "tribal_to_hub"
    return "other"


def _infer_tribal(src_lang, tgt_lang):
    st = _infer_lang(src_lang, config.TRIBAL_LANGS)
    tt = _infer_lang(tgt_lang, config.TRIBAL_LANGS)
    return st or tt or "hub"


# ======================================================
# BUILD PATHS
# ======================================================
variant = config.MODEL_VARIANT.replace(".", "_")
suffix = "lora" if config.USE_LORA else "full"

model_output_dir = os.path.join(
    config.OUTPUT_DIR,
    f"{variant}_{suffix}"
)

os.makedirs(model_output_dir, exist_ok=True)

inp = os.path.join(config.DATA_DIR, "val.csv")

out = os.path.join(
    model_output_dir,
    "inference_output.csv"
)

if not os.path.exists(inp):
    raise FileNotFoundError(f"❌ Input file not found:\n{inp}")

# ======================================================
# LOAD CSV
# ======================================================
df = pd.read_csv(inp)

# DEBUG MODE SUPPORT
if config.MODE == "debug":
    print(f"\n🐞 DEBUG MODE ENABLED")
    print(f"Using first {config.DEBUG_ROWS} rows only\n")
    df = df.head(config.DEBUG_ROWS)

required_cols = {
    "source_sentence",
    "target_sentence",
    "source_lang",
    "target_lang",
}

if not required_cols.issubset(df.columns):
    raise ValueError(
        f"❌ Input CSV must contain columns:\n{required_cols}"
    )

# ======================================================
# INFERENCE LOOP
# ======================================================
rows = []

iterator = tqdm(
    df.iterrows(),
    total=len(df),
    desc="Running Inference",
    dynamic_ncols=True,
)

for idx, r in iterator:

    src = str(r["source_sentence"])
    tgt = str(r["target_sentence"])
    sl = str(r["source_lang"])
    tl = str(r["target_lang"])

    direction = _infer_route(sl, tl)
    tribal = _infer_tribal(sl, tl)

    pred = translate(src, sl, tl)

    if config.MODE == "debug":
        print("\n------------------------------")
        print("SRC :", src)
        print("PRED:", pred)
        print("------------------------------\n")

    rows.append({
        "Tribal": tribal,
        "Direction": direction,
        "Source Language": sl,
        "Target Language": tl,
        "Source Sentence": src,
        "Target Sentence (Actual)": tgt,
        "Target Sentence (Predicted)": pred,
    })

# ======================================================
# SAVE
# ======================================================
pd.DataFrame(rows).to_csv(out, index=False)

print("\n✅ Inference completed.")
print(f"📁 File saved at: {out}\n")
