# eval_csv.py
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load
from tqdm import tqdm

from config import *
from dataset_utils import get_test_dataset_path

bleu = load("bleu")
chrf = load("chrf")


def _pretty_name(lang: str):
    return lang.capitalize()


def _safe_bleu(pred, ref):
    try:
        if pred.strip() == "" or ref.strip() == "":
            return 0.0
        return bleu.compute(
            predictions=[pred],
            references=[[ref]],
        )["bleu"] * 100
    except Exception:
        return 0.0


def _safe_chrf(pred, ref):
    try:
        if pred.strip() == "" or ref.strip() == "":
            return 0.0
        return chrf.compute(
            predictions=[pred],
            references=[ref],
        )["score"]
    except Exception:
        return 0.0


def evaluate_csv():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for tribal in TRIBAL_LANGS:
        hubs = HUB_LANGS_BHILI if tribal == "bhili" else HUB_LANGS_COMMON
        hp = HYPERPARAMS[tribal]

        for hub in hubs:
            for src, tgt in [(hub, tribal), (tribal, hub)]:
                direction_key = f"{src}_{tgt}"

                if DIRECTION_CONFIG.get(tribal, {}).get(direction_key, True) is False:
                    continue

                df = pd.read_csv(get_test_dataset_path(tribal))
                if DEBUG_MODE:
                    df = df.head(DEBUG_TEST_ROWS)

                src_col = LANGUAGE_COLUMN_MAP[tribal][src]
                tgt_col = LANGUAGE_COLUMN_MAP[tribal][tgt]

                model_name = MODEL_CHOICE[tribal].split("/")[-1]
                tag = "lora" if USE_LORA[tribal] else "normal"
                model_dir = os.path.join(
                    MODEL_SAVE_DIR,
                    f"{model_name}_{tag}",
                    tribal,
                    direction_key,
                )

                if not os.path.exists(model_dir):
                    print(f"[Skip] Model not found for {tribal} {src}->{tgt}")
                    continue

                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

                src_name = _pretty_name(src)
                tgt_name = _pretty_name(tgt)

                rows = []

                iterator = tqdm(
                    df.iterrows(),
                    total=len(df),
                    desc=f"[Eval] {tribal} {src}->{tgt}",
                )

                for _, row in iterator:
                    source_text = str(row[src_col])
                    target_text = str(row[tgt_col])

                    inputs = tokenizer(source_text, return_tensors="pt").to(device)
                    output = model.generate(
                        **inputs,
                        num_beams=hp["beam_width"],
                        max_length=hp["max_length"],
                    )
                    pred = tokenizer.decode(output[0], skip_special_tokens=True)

                    bleu_score = _safe_bleu(pred, target_text)
                    chrf_score = _safe_chrf(pred, target_text)

                    rows.append([
                        source_text,
                        target_text,
                        pred,
                        bleu_score,
                        chrf_score,
                    ])

                out = pd.DataFrame(
                    rows,
                    columns=[
                        src_name,
                        f"Actual {tgt_name}",
                        f"Predicted {tgt_name}",
                        "BLEU Score (100)",
                        "CHRF++ Score (100)",
                    ],
                )

                save_dir = os.path.join(
                    PROJECT_DIR,
                    "eval-infer",
                    f"{model_name}_{tag}",
                    tribal,
                    direction_key,
                )
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, "eval.csv")
                out.to_csv(save_path, index=False)
                print(f"[Saved] {save_path}")


if __name__ == "__main__":
    evaluate_csv()
