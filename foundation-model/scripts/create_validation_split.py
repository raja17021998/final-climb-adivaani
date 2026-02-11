import os
import pandas as pd
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

UNIFIED_TRAIN_PATH = "/home/user/Desktop/Shashwat/final-climb/datasets/unified/small_train_10k.csv"
VAL_OUTPUT_PATH = "/home/user/Desktop/Shashwat/final-climb/validation-data/unified/small-val_10k.csv"

VAL_RATIO = 0.1          # 10% validation
RANDOM_STATE = 42

os.makedirs(os.path.dirname(VAL_OUTPUT_PATH), exist_ok=True)

# ============================================================
# MAIN LOGIC
# ============================================================

def main():
    df = pd.read_csv(UNIFIED_TRAIN_PATH)

    assert {
        "source_sentence",
        "target_sentence",
        "source_lang",
        "target_lang"
    }.issubset(df.columns), "Unified CSV schema mismatch"

    # --------------------------------------------------------
    # Group rows by translation direction
    # --------------------------------------------------------
    direction_groups = defaultdict(list)

    for idx, row in df.iterrows():
        key = (row["source_lang"], row["target_lang"])
        direction_groups[key].append(idx)

    val_indices = []

    # --------------------------------------------------------
    # Sample proportionally per direction
    # --------------------------------------------------------
    for (src_lang, tgt_lang), indices in direction_groups.items():
        n_total = len(indices)
        n_val = max(1, int(n_total * VAL_RATIO))

        sampled = (
            pd.Series(indices)
            .sample(n=n_val, random_state=RANDOM_STATE)
            .tolist()
        )

        val_indices.extend(sampled)

        print(
            f"Direction {src_lang} → {tgt_lang} | "
            f"Total: {n_total} | Val: {n_val}"
        )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------
    val_df = df.loc[val_indices].reset_index(drop=True)
    train_df = df.drop(index=val_indices).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    val_df.to_csv(VAL_OUTPUT_PATH, index=False, encoding="utf-8")
    train_df.to_csv(UNIFIED_TRAIN_PATH, index=False, encoding="utf-8")

    print("================================================")
    print("Validation split created")
    print("Train size:", len(train_df))
    print("Val size:", len(val_df))
    print("Saved val to:", VAL_OUTPUT_PATH)
    print("================================================")


if __name__ == "__main__":
    main()
