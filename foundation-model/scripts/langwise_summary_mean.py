# # langwise_summary_mean.py
# import os
# import pandas as pd
# import config

# # ======================================================
# # BUILD PATHS
# # ======================================================
# variant = config.MODEL_VARIANT.replace(".", "_")
# suffix = "lora" if config.USE_LORA else "full"

# model_output_dir = os.path.join(
#     config.OUTPUT_DIR,
#     f"{variant}_{suffix}"
# )

# eval_dir = os.path.join(model_output_dir, "eval")

# inp = os.path.join(
#     eval_dir,
#     f"langwise_eval_{variant}_{suffix}.csv"
# )

# out = os.path.join(
#     eval_dir,
#     f"langwise_summary_mean_{variant}_{suffix}.csv"
# )

# # ======================================================
# # LOAD
# # ======================================================
# if not os.path.exists(inp):
#     raise FileNotFoundError(f"❌ File not found:\n{inp}")

# df = pd.read_csv(inp)

# # ======================================================
# # BUILD Tribal_Direction KEY
# # ======================================================
# df["Tribal_Direction"] = (
#     df["Tribal"].astype(str) + "_" + df["Direction"].astype(str)
# )

# # ======================================================
# # MEAN AGGREGATION
# # ======================================================
# summary = (
#     df.groupby("Tribal_Direction")
#       .agg(
#           **{
#               "Corpus BLEU": ("BLEU (sentence)", "mean"),
#               "Corpus chrF++": ("chrF++ (sentence)", "mean"),
#           }
#       )
#       .reset_index()
#       .sort_values("Tribal_Direction")
# )

# # ======================================================
# # SAVE
# # ======================================================
# summary.to_csv(out, index=False)

# print("\n✅ Langwise summary generated (MEAN of sentence metrics).")
# print(f"📁 File saved at: {out}\n")


# langwise_summary_mean.py

import os
import pandas as pd
import evaluate
import config

# ======================================================
# PATH SETUP
# ======================================================
variant = config.MODEL_VARIANT.replace(".", "_")
suffix = "lora" if config.USE_LORA else "full"

model_output_dir = os.path.join(
    config.OUTPUT_DIR,
    f"{variant}_{suffix}"
)

eval_dir = os.path.join(model_output_dir, "eval")

input_csv = os.path.join(
    eval_dir,
    f"langwise_eval_{variant}_{suffix}.csv"
)

output_csv = os.path.join(
    eval_dir,
    f"langwise_summary_{variant}_{suffix}.csv"
)

# ======================================================
# LOAD DATA
# ======================================================
if not os.path.exists(input_csv):
    raise FileNotFoundError(f"❌ Eval file not found: {input_csv}")

df = pd.read_csv(input_csv)

if df.empty:
    raise ValueError("❌ Eval CSV is empty.")

# ======================================================
# LOAD METRICS
# ======================================================
bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

# ======================================================
# BUILD EXACT LANGUAGE PAIR
# ======================================================
df["Exact_Direction"] = (
    df["Source Language"].astype(str) +
    "_to_" +
    df["Target Language"].astype(str)
)

df["Tribal_Direction"] = (
    df["Tribal"].astype(str) +
    "_" +
    df["Exact_Direction"]
)

# ======================================================
# GROUP + CORPUS METRICS
# ======================================================
results = []

for group_name, group_df in df.groupby("Tribal_Direction"):

    predictions = group_df["Target Sentence (Predicted)"].astype(str).tolist()
    references = group_df["Target Sentence (Actual)"].astype(str).tolist()

    # BLEU expects list of list references
    references_bleu = [[ref] for ref in references]

    try:
        bleu_score = bleu_metric.compute(
            predictions=predictions,
            references=references_bleu,
        )["bleu"] * 100

        chrf_score = chrf_metric.compute(
            predictions=predictions,
            references=references,
        )["score"]

    except Exception:
        bleu_score = 0.0
        chrf_score = 0.0

    results.append({
        "Tribal_Direction": group_name,
        "Corpus BLEU": bleu_score,
        "Corpus chrF++": chrf_score,
        "Num Samples": len(group_df),
    })

# ======================================================
# SAVE
# ======================================================
summary_df = pd.DataFrame(results)
summary_df = summary_df.sort_values("Tribal_Direction")

summary_df.to_csv(output_csv, index=False)

print("\n✅ Langwise summary (corpus-level) complete.")
print(f"📁 Saved at: {output_csv}\n")

