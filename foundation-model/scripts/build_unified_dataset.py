import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATASETS_DIR = "/home/kshhorizon/data/final-climb-shashwat-do-not-delete/datasets"
UNIFIED_DIR = os.path.join(DATASETS_DIR, "foundation-model")
TEST_DIR = os.path.join(DATASETS_DIR, "test")

os.makedirs(UNIFIED_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

RANDOM_STATE = 42
CSV_TEST_RATIO = 0.10
UNIFIED_VAL_RATIO = 0.10

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

        train_df, test_df = train_test_split(
            df,
            test_size=CSV_TEST_RATIO,
            random_state=RANDOM_STATE,
        )

        test_df.to_csv(
            os.path.join(TEST_DIR, f"test_{fname}"),
            index=False,
            encoding="utf-8",
        )

        unified_rows.extend(build_pairs(train_df, tribal, others, fname))

    unified_df = (
        pd.DataFrame(unified_rows)
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    print(f"Total unified rows (before split): {len(unified_df)}")

    train_df, val_df = train_test_split(
        unified_df,
        test_size=UNIFIED_VAL_RATIO,
        random_state=RANDOM_STATE,
    )

    train_df.to_csv(os.path.join(UNIFIED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(UNIFIED_DIR, "val.csv"), index=False)

    print(f"Final train rows: {len(train_df)}")
    print(f"Final validation rows: {len(val_df)}")
    print("Unified train/val created")

if __name__ == "__main__":
    main()
