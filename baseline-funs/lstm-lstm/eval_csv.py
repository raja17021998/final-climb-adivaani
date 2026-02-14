# eval_csv.py
import pandas as pd
from evaluate import load
from infer import infer
import config as cfg

bleu = load("bleu")
chrf = load("chrf")

def evaluate_file(lang, direction, file):
    src, tgt = direction.split("_")
    df = pd.read_csv(file)

    if cfg.DEBUG_MODE:
        df = df.head(cfg.DEBUG_TEST_ROWS)

    src_col = cfg.LANGUAGE_COLUMN_MAP[lang][src]
    tgt_col = cfg.LANGUAGE_COLUMN_MAP[lang][tgt]

    preds, bleus, chrfs = [], [], []
    for s, t in zip(df[src_col], df[tgt_col]):
        p = infer(s, src, tgt)
        preds.append(p)
        bleus.append(bleu.compute(predictions=[p], references=[[t]])["bleu"] * 100)
        chrfs.append(chrf.compute(predictions=[p], references=[[t]])["score"])

    df["Predicted Target Language"] = preds
    df["BLEU Score"] = bleus
    df["CHRF++ Score"] = chrfs
    print(df)
