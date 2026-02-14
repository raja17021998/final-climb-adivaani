# infer_csv.py
import pandas as pd
from infer import infer
import config as cfg

def run(lang, direction, file):
    src, tgt = direction.split("_")
    df = pd.read_csv(file)

    if cfg.DEBUG_MODE:
        df = df.head(cfg.DEBUG_TEST_ROWS)

    src_col = cfg.LANGUAGE_COLUMN_MAP[lang][src]
    tgt_col = cfg.LANGUAGE_COLUMN_MAP[lang][tgt]

    preds = []
    for s in df[src_col]:
        preds.append(infer(s, src, tgt))

    df["Predicted Target Language"] = preds
    print(df[[src_col, tgt_col, "Predicted Target Language"]])
