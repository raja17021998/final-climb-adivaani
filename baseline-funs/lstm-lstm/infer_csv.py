# infer_csv.py
import os
import argparse
import pandas as pd
from tqdm import tqdm
from infer import infer
import config as cfg

TEST_FILE_MAP = {
    "bhili": "test_Bhi_Hin_Mar_Guj_Eng.csv",
    "mundari": "test_Mun_Hin_Eng.csv",
    "gondi": "test_Gon_Hin_Eng.csv",
    "santali": "test_San_Hin_Eng.csv",
    "kui": "test_Kui_Hin_Eng.csv",
    "garo": "test_Garo_Hin_Eng.csv",
}

def get_active_directions(lang):
    directions = []
    for direction, flag in cfg.DIRECTION_CONFIG[lang].items():
        if not flag:
            continue
        limit = cfg.DIRECTION_DATA_LIMIT.get(lang, {}).get(direction, None)
        if limit is not None:
            directions.append(direction)
    return directions

def get_default_test_file(lang):
    return os.path.join(cfg.TEST_DATA_DIR, TEST_FILE_MAP[lang])

def run_inference(lang, direction, file):
    src, tgt = direction.split("_")
    df = pd.read_csv(file)

    limit = cfg.DIRECTION_DATA_LIMIT.get(lang, {}).get(direction, None)
    if limit is not None:
        df = df.head(limit)

    if cfg.DEBUG_MODE:
        df = df.head(min(len(df), cfg.DEBUG_TEST_ROWS))

    src_col = cfg.LANGUAGE_COLUMN_MAP[lang][src]
    tgt_col = cfg.LANGUAGE_COLUMN_MAP[lang][tgt]

    rows = []
    iterator = tqdm(
        zip(df[src_col], df[tgt_col]),
        total=len(df),
        desc=f"Infer {lang}-{direction}"
    )

    for s, t in iterator:
        p = infer(s, src, tgt)
        rows.append({
            src_col: s,
            f"Actual {tgt_col}": t,
            f"Predicted {tgt_col}": p,
        })

    out_df = pd.DataFrame(rows)

    save_dir = os.path.join(cfg.BASE_DIR, "baseline-funs/lstm-lstm/eval-infer", lang, direction)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "infer.csv")
    out_df.to_csv(save_path, index=False)

    print(out_df)
    print(f"\nSaved to: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="bhili", help="tribal language key")
    parser.add_argument("--file", type=str, default=None, help="optional test CSV path")
    args = parser.parse_args()

    lang = args.lang
    test_file = args.file if args.file else get_default_test_file(lang)

    active_dirs = get_active_directions(lang)
    for direction in active_dirs:
        run_inference(lang, direction, test_file)

if __name__ == "__main__":
    main()
