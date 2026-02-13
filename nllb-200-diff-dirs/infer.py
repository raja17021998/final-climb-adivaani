# infer.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import *

def infer(sentence: str, src_lang: str, tgt_lang: str) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tribal = src_lang if src_lang in TRIBAL_LANGS else tgt_lang
    hp = HYPERPARAMS[tribal]

    model_name = MODEL_CHOICE[tribal].split("/")[-1]
    tag = "lora" if USE_LORA[tribal] else "normal"
    model_dir = os.path.join(MODEL_SAVE_DIR, f"{model_name}_{tag}", tribal, f"{src_lang}_{tgt_lang}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    output = model.generate(**inputs, num_beams=hp["beam_width"], max_length=hp["max_length"])
    return tokenizer.decode(output[0], skip_special_tokens=True)
