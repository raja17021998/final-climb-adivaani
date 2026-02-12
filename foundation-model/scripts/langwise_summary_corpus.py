# langwise_summary_corpus.py
import os
import pandas as pd
import sacrebleu
import config

# ======================================================
# BUILD PATHS
# ======================================================
variant = config.MODEL_VARIANT.replace(".", "_")
suffix = "lora" if config.USE_LORA else "full"

model_output_dir = os.path.join(
    config.OUTPUT_DIR,
    f"{variant}_{suffix}"
)

eval_dir = os.path.join(model_output_dir, "eval")

inp = os.path.join(
    eval_dir,
    f"langwise_eval_{variant}_{suffix}.csv"
)

out = os.path.join(
    eval_dir,
    f"langwise_summary_corpus_{variant}_{suffix}.csv"
)

# ======================================================
# LOAD
# ======================================================
if not os.path.exists(inp):
    raise FileNotFoundError(f"❌ File not found:\n{inp}")

df = pd.read_csv(inp)

# ======================================================
# BUILD Tribal_Direction KEY
# ======================================================
df["Tribal_Direction"] = (
    df["Tribal"].astype(str) + "_" + df["Direction"].astype(str)
)

# ======================================================
# TRUE CORPUS METRICS
# ======================================================
rows = []

for key, group in df.groupby("Tribal_Direction"):

    refs = group["Target Sentence (Actual)"].astype(str).tolist()
    hyps = group["Target Sentence (Predicted)"].astype(str).tolist()

    # Remove completely empty predictions (important)
    cleaned = [
        (h, r) for h, r in zip(hyps, refs)
        if h.strip() != ""
    ]

    if len(cleaned) == 0:
        bleu = 0.0
        chrf = 0.0
    else:
        hyps_clean, refs_clean = zip(*cleaned)

        bleu = sacrebleu.corpus_bleu(
            hyps_clean,
            [refs_clean]
        ).score

        chrf = sacrebleu.corpus_chrf(
            hyps_clean,
            [refs_clean]
        ).score

    rows.append({
        "Tribal_Direction": key,
        "Corpus BLEU": bleu,
        "Corpus chrF++": chrf,
    })

summary = (
    pd.DataFrame(rows)
      .sort_values("Tribal_Direction")
)

# ======================================================
# SAVE
# ======================================================
summary.to_csv(out, index=False)

print("\n✅ Langwise summary generated (TRUE CORPUS metrics).")
print(f"📁 File saved at: {out}\n")
