# infer.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from config import *


def infer(sentence: str, src_lang: str, tgt_lang: str) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tribal = src_lang if src_lang in TRIBAL_LANGS else tgt_lang
    hp = HYPERPARAMS[tribal]

    direction_key = f"{src_lang}_{tgt_lang}"

    if DIRECTION_CONFIG.get(tribal, {}).get(direction_key, True) is False:
        raise ValueError(f"Inference for direction '{direction_key}' is disabled in config.")

    model_name = MODEL_CHOICE[tribal].split("/")[-1]
    tag = "lora" if USE_LORA[tribal] else "normal"

    model_dir = os.path.join(
        MODEL_SAVE_DIR,
        f"{model_name}_{tag}",
        tribal,
        direction_key,
    )

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found for direction '{direction_key}' at {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

    # tqdm used for single sentence as uniform interface
    for _ in tqdm(range(1), desc=f"[Infer] {src_lang}->{tgt_lang}"):
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            num_beams=hp["beam_width"],
            max_length=hp["max_length"],
        )
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)

    return prediction
