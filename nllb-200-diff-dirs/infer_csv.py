# infer_csv.py
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import *
from dataset_utils import get_test_dataset_path

def infer_csv():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for tribal in TRIBAL_LANGS:
        hubs = HUB_LANGS_BHILI if tribal == "bhili" else HUB_LANGS_COMMON
        hp = HYPERPARAMS[tribal]
        for hub in hubs:
            for src, tgt in [(hub, tribal), (tribal, hub)]:
                df = pd.read_csv(get_test_dataset_path(tribal))
                if DEBUG_MODE:
                    df = df.head(DEBUG_TEST_ROWS)

                src_col = LANGUAGE_COLUMN_MAP[tribal][src]
                tgt_col = LANGUAGE_COLUMN_MAP[tribal][tgt]

                model_name = MODEL_CHOICE[tribal].split("/")[-1]
                tag = "lora" if USE_LORA[tribal] else "normal"
                model_dir = os.path.join(MODEL_SAVE_DIR, f"{model_name}_{tag}", tribal, f"{src}_{tgt}")

                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

                preds = []
                for _, row in df.iterrows():
                    inputs = tokenizer(str(row[src_col]), return_tensors="pt").to(device)
                    output = model.generate(**inputs, num_beams=hp["beam_width"], max_length=hp["max_length"])
                    preds.append(tokenizer.decode(output[0], skip_special_tokens=True))

                out = pd.DataFrame({
                    "Source Language": df[src_col],
                    "Actual Target Language": df[tgt_col],
                    "Predicted Target Language": preds,
                })
                out.to_csv(f"infer_{tribal}_{src}_{tgt}.csv", index=False)

if __name__ == "__main__":
    infer_csv()
