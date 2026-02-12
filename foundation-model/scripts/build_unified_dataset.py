import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import config

# ======================================================
# PATHS
# ======================================================
DATASETS_DIR = os.path.dirname(config.DATA_DIR)
UNIFIED_DIR = config.DATA_DIR

os.makedirs(UNIFIED_DIR, exist_ok=True)

RANDOM_STATE = config.SEED

TRIBAL_LANGS = {"Bhili", "Garo", "Gondi", "Kui", "Mundari", "Santali"}

LANGUAGE_TAGS = {
    "Bhili": "bhi_Deva",
    "Garo": "gar_Latn",
    "Gondi": "gon_Deva",
    "Kui": "kui_Orya",
    "Mundari": "mun_Deva",
    "Santali": "san_Olck",
    "Hindi": "hin_Deva",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "English": "eng_Latn",
}

CSV_FILES = [
    "Bhi_Hin_Mar_Guj_Eng.csv",
    "Garo_Hin_Eng.csv",
    "Gon_Hin_Eng.csv",
    "Kui_Hin_Eng.csv",
    "Mun_Hin_Eng.csv",
    "San_Hin_Eng.csv",
]

# ======================================================
# HELPERS
# ======================================================
def extract_langs(df):
    return [c for c in df.columns if c != "Unique_ID"]

def clean_df(df, cols):
    df = df.dropna(subset=cols)
    for c in cols:
        df = df[df[c].astype(str).str.strip().ne("")]
    return df.reset_index(drop=True)

def build_pairs(df, tribal, others, dataset):
    rows = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Building pairs [{dataset}]",
        leave=False,
    ):
        for lang in others:
            rows.append({
                "source_sentence": row[tribal],
                "target_sentence": row[lang],
                "source_lang": LANGUAGE_TAGS[tribal],
                "target_lang": LANGUAGE_TAGS[lang],
                "dataset": dataset,
            })
            rows.append({
                "source_sentence": row[lang],
                "target_sentence": row[tribal],
                "source_lang": LANGUAGE_TAGS[lang],
                "target_lang": LANGUAGE_TAGS[tribal],
                "dataset": dataset,
            })

    return rows

# ======================================================
# MAIN
# ======================================================
def main():

    unified_rows = []

    for fname in tqdm(CSV_FILES, desc="Processing CSV files"):

        df = pd.read_csv(os.path.join(DATASETS_DIR, fname))
        langs = extract_langs(df)

        tribal = list(set(langs) & TRIBAL_LANGS)
        assert len(tribal) == 1
        tribal = tribal[0]

        others = [l for l in langs if l != tribal]

        df = clean_df(df, [tribal] + others)

        unified_rows.extend(
            build_pairs(df, tribal, others, fname)
        )

    # ======================================================
    # BUILD UNIFIED DF
    # ======================================================
    unified_df = pd.DataFrame(unified_rows)

    # 🔥 FULL RANDOM SHUFFLE BEFORE SPLIT
    unified_df = (
        unified_df
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    total = len(unified_df)
    print(f"\nTotal unified rows: {total}")

    # ======================================================
    # 🔥 PRINT TOTAL ROWS PER DIRECTION (ADDED)
    # ======================================================
    unified_df["direction"] = (
        unified_df["source_lang"] +
        "_to_" +
        unified_df["target_lang"]
    )

    direction_counts = (
        unified_df["direction"]
        .value_counts()
        .sort_index()
    )

    print("\nTotal rows per direction:")
    for direction, count in direction_counts.items():
        print(f"{direction:<30} {count}")

    print("\n")

    # ======================================================
    # TRAIN / VAL / TEST SPLIT FROM CONFIG
    # ======================================================
    train_ratio = config.TRAIN_RATIO
    val_ratio = config.VAL_RATIO
    test_ratio = config.TEST_RATIO

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    train_df, temp_df = train_test_split(
        unified_df,
        test_size=(1.0 - train_ratio),
        random_state=RANDOM_STATE,
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=RANDOM_STATE,
    )

    # Remove helper column before saving
    train_df = train_df.drop(columns=["direction"])
    val_df = val_df.drop(columns=["direction"])
    test_df = test_df.drop(columns=["direction"])

    # ======================================================
    # SAVE
    # ======================================================
    train_df.to_csv(os.path.join(UNIFIED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(UNIFIED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(UNIFIED_DIR, "test.csv"), index=False)

    print(f"Final train rows: {len(train_df)}")
    print(f"Final validation rows: {len(val_df)}")
    print(f"Final test rows: {len(test_df)}")
    print("Unified train/val/test created successfully.")

if __name__ == "__main__":
    main()
