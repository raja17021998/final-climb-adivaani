# langwise_summary_mean.py
import os
import pandas as pd
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
    f"langwise_summary_mean_{variant}_{suffix}.csv"
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
# MEAN AGGREGATION
# ======================================================
summary = (
    df.groupby("Tribal_Direction")
      .agg(
          **{
              "Corpus BLEU": ("BLEU (sentence)", "mean"),
              "Corpus chrF++": ("chrF++ (sentence)", "mean"),
          }
      )
      .reset_index()
      .sort_values("Tribal_Direction")
)

# ======================================================
# SAVE
# ======================================================
summary.to_csv(out, index=False)

print("\n✅ Langwise summary generated (MEAN of sentence metrics).")
print(f"📁 File saved at: {out}\n")
