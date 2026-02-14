# infer.py
import os
import argparse
import torch
import sentencepiece as spm
import config as cfg
from train_ddp import Seq2Seq, load_special_tokens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp = spm.SentencePieceProcessor(model_file=cfg.TOKENIZER_MODEL)
special_tokens = load_special_tokens(sp)

PAD_ID = special_tokens.get("<pad>") or 0
BOS_ID = special_tokens.get("<s>") or 1
EOS_ID = special_tokens.get("</s>") or 2

BEAM_WIDTH = cfg.TRAINING_PARAMS.get("beam_width", 3)
MAX_LEN = cfg.TRAINING_PARAMS["max_length"]

def load_model(lang, direction):
    model_path = os.path.join(cfg.MODEL_SAVE_DIR, lang, direction, "best_model.pt")
    model = Seq2Seq(PAD_ID, BOS_ID).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def encode(sentence):
    ids = sp.encode(sentence, out_type=int)
    return torch.tensor([BOS_ID] + ids + [EOS_ID], dtype=torch.long).unsqueeze(0).to(DEVICE)

def decode(ids):
    tokens = [i for i in ids if i not in {PAD_ID, BOS_ID, EOS_ID}]
    return sp.decode(tokens)

def beam_search_decode(model, src_tensor, beam_width, max_len):
    _, hidden = model.encoder(src_tensor)

    beams = [(torch.tensor([BOS_ID], device=DEVICE), hidden, 0.0)]
    completed = []

    for _ in range(max_len):
        new_beams = []
        for seq, hidden_state, score in beams:
            inp = seq[-1].unsqueeze(0)
            out, new_hidden = model.decoder(inp, hidden_state)
            log_probs = torch.log_softmax(out, dim=1)
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)

            for k in range(beam_width):
                token = topk_ids[0, k].item()
                new_score = score + topk_log_probs[0, k].item()
                new_seq = torch.cat([seq, torch.tensor([token], device=DEVICE)])

                if token == EOS_ID:
                    completed.append((new_seq, new_score))
                else:
                    new_beams.append((new_seq, new_hidden, new_score))

        beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]
        if not beams:
            break

    if completed:
        best_seq = sorted(completed, key=lambda x: x[1], reverse=True)[0][0]
    else:
        best_seq = beams[0][0]

    return best_seq.tolist()

def infer(sentence: str, src_lang: str, tgt_lang: str) -> str:
    direction = f"{src_lang}_{tgt_lang}"
    lang_key = tgt_lang if tgt_lang in cfg.TRIBAL_LANGS else src_lang
    model = load_model(lang_key, direction)

    src_tensor = encode(sentence)
    pred_ids = beam_search_decode(model, src_tensor, BEAM_WIDTH, MAX_LEN)

    return decode(pred_ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument("--src_lang", type=str, default="english")
    parser.add_argument("--tgt_lang", type=str, default="bhili")
    args = parser.parse_args()

    output = infer(args.sentence, args.src_lang, args.tgt_lang)
    print(output)

if __name__ == "__main__":
    main()
