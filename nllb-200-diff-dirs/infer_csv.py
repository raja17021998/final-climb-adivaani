# infer_csv.py
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from config import *
from dataset_utils import get_test_dataset_path


def _pretty_name(lang: str):
    return lang.capitalize()


def infer_csv():
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
                    desc=f"[Infer CSV] {tribal} {src}->{tgt}",
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

                    rows.append([source_text, target_text, pred])

                out = pd.DataFrame(
                    rows,
                    columns=[
                        src_name,
                        f"Actual {tgt_name}",
                        f"Predicted {tgt_name}",
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

                save_path = os.path.join(save_dir, "infer.csv")
                out.to_csv(save_path, index=False)
                print(f"[Saved] {save_path}")


if __name__ == "__main__":
    infer_csv()


