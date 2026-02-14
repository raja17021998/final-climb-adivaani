# infer.py
import os
import torch
import sentencepiece as spm
import config as cfg
from text_protection import protect_text, restore_text
from train_ddp import Seq2Seq

sp = spm.SentencePieceProcessor(model_file=cfg.TOKENIZER_MODEL)

def load_model(lang, direction):
    model = Seq2Seq()
    path = os.path.join(cfg.MODEL_SAVE_DIR, lang, direction, "model.pt")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def infer(sentence: str, src_lang: str, tgt_lang: str) -> str:
    direction = f"{src_lang}_{tgt_lang}"
    model = load_model(tgt_lang if tgt_lang in cfg.TRIBAL_LANGS else src_lang, direction)
    sent, mapping = protect_text(sentence)
    ids = sp.encode(sent, out_type=int)
    inp = torch.tensor(ids).unsqueeze(0)
    with torch.no_grad():
        enc_out, hidden = model.encoder(inp)
        token = torch.tensor([1])
        outputs=[]
        for _ in range(cfg.TRAINING_PARAMS["max_length"]):
            out, hidden = model.decoder(token, hidden)
            token = out.argmax(1)
            if token.item()==2: break
            outputs.append(token.item())
    text = sp.decode(outputs)
    return restore_text(text, mapping)
