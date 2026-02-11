import argparse
import sys
import pandas as pd

# ======================================================
# FORCE IMMEDIATE STDOUT FLUSH (IMPORTANT FOR BASH)
# ======================================================
sys.stdout.reconfigure(line_buffering=True)

# ======================================================
# ARGS
# ======================================================
parser = argparse.ArgumentParser(description="Build language-wise summary from sentence-level evaluation CSV")
parser.add_argument(
    "--input_csv",
    required=True,
    help="Sentence-level evaluation CSV from langwise_eval.py",
)
parser.add_argument(
    "--output_csv",
    required=True,
    help="Path to save language-wise summary CSV",
)

args = parser.parse_args()

print("🚀 Starting langwise_summary.py", flush=True)
print(f"📥 Input CSV  : {args.input_csv}", flush=True)
print(f"📤 Output CSV : {args.output_csv}", flush=True)

# ======================================================
# LOAD SENTENCE-LEVEL RESULTS
# ======================================================
try:
    df = pd.read_csv(args.input_csv)
except Exception as e:
    raise RuntimeError(f"❌ Failed to read input CSV: {args.input_csv}") from e

print(f"📄 Loaded {len(df)} sentence-level rows", flush=True)

# ======================================================
# VALIDATE REQUIRED COLUMNS
# ======================================================
required_cols = [
    "Source Language",
    "Target Language",
    "BLEU (sentence)",
    "chrF++ (sentence)",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(
        f"❌ Missing required columns in input CSV: {missing}\n"
        f"Found columns: {list(df.columns)}"
    )

print("✅ Required columns present", flush=True)

# ======================================================
# BUILD DIRECTION COLUMN
# ======================================================
df["Direction"] = df["Source Language"] + " -> " + df["Target Language"]

print(f"🔁 Built Direction column ({df['Direction'].nunique()} unique directions)", flush=True)

# ======================================================
# AGGREGATE
# ======================================================
print("📊 Aggregating sentence-level metrics...", flush=True)

summary = (
    df.groupby("Direction")
    .agg(
        Num_Sentences=("Direction", "count"),
        Avg_BLEU=("BLEU (sentence)", "mean"),
        Avg_chrFpp=("chrF++ (sentence)", "mean"),
    )
    .reset_index()
)

# Optional: sort by number of sentences (descending)
summary = summary.sort_values("Num_Sentences", ascending=False)

print(f"📈 Aggregation complete: {len(summary)} language directions", flush=True)

# ======================================================
# PREVIEW TOP DIRECTIONS (VISIBILITY)
# ======================================================
print("\n🔍 Top language directions by volume:", flush=True)
for _, row in summary.head(10).iterrows():
    print(
        f"  {row['Direction']:<35} | "
        f"N={int(row['Num_Sentences']):<5} | "
        f"BLEU={row['Avg_BLEU']:.2f} | "
        f"chrF++={row['Avg_chrFpp']:.2f}",
        flush=True,
    )

# ======================================================
# SAVE
# ======================================================
summary.to_csv(args.output_csv, index=False)

print("\n✅ Language-wise summary saved successfully", flush=True)
print(f"📁 Path: {args.output_csv}", flush=True)
print("🏁 langwise_summary.py completed", flush=True)
