import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import sentencepiece as spm
import evaluate
from tqdm import tqdm
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("sacrebleu").setLevel(logging.ERROR)

# ===================== CONFIG =====================

class Config:
    ROOT_DIR = "/home/user/Desktop/Shashwat/final-climb/baseline-funs"
    DATASET_DIR = "/home/user/Desktop/Shashwat/final-climb/datasets"
    SPM_PATH = os.path.join(ROOT_DIR, "joint_spm.model")
    MODEL_PATH = os.path.join(ROOT_DIR, "lstm-attn", "Shared_Model", "shared_best.pt")

    LANGS = ["bhili", "gondi", "kui", "mundari"]
    CSV_FILES = {
        "bhili": "Hin_Bhi_Mar_Guj.csv",
        "gondi": "Hin_Gon.csv",
        "kui": "Hin_Kui.csv",
        "mundari": "Hin_Mun.csv"
    }

    # Per-language evaluation limits (None = full CSV)
    EVAL_LIMITS = {
        "bhili": 1000,
        "gondi": 1000,
        "kui": 1000,
        "mundari": 1000
    }

    VOCAB_SIZE = 8000
    EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 1
    MAX_LEN = 60

    BOS_ID = 1
    EOS_ID = 2

# ===================== MODEL =====================

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.EMB_DIM)
        self.rnn = nn.LSTM(
            Config.EMB_DIM,
            Config.HID_DIM,
            Config.N_LAYERS,
            bidirectional=True
        )
        self.fc_h = nn.Linear(Config.HID_DIM * 2, Config.HID_DIM)
        self.fc_c = nn.Linear(Config.HID_DIM * 2, Config.HID_DIM)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (h, c) = self.rnn(embedded)

        h = torch.tanh(self.fc_h(torch.cat((h[-2], h[-1]), dim=1)))
        c = torch.tanh(self.fc_c(torch.cat((c[-2], c[-1]), dim=1)))

        return outputs, (h.unsqueeze(0), c.unsqueeze(0))


class SharedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(Config.HID_DIM * 3, Config.HID_DIM)
        self.v = nn.Linear(Config.HID_DIM, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(src_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return torch.softmax(attention, dim=1)


class TribalDecoder(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.EMB_DIM)
        self.rnn = nn.LSTM(
            Config.HID_DIM * 2 + Config.EMB_DIM,
            Config.HID_DIM,
            Config.N_LAYERS
        )
        self.fc_out = nn.Linear(
            Config.HID_DIM * 3 + Config.EMB_DIM,
            Config.VOCAB_SIZE
        )

    def forward(self, input, h, c, encoder_outputs):
        embedded = self.embedding(input.unsqueeze(0))
        a = self.attention(h, encoder_outputs).unsqueeze(1)

        enc_outs = encoder_outputs.transpose(0, 1)
        weighted = torch.bmm(a, enc_outs).transpose(0, 1)

        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (h, c) = self.rnn(rnn_input, (h, c))

        pred = self.fc_out(
            torch.cat(
                (output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)),
                dim=1
            )
        )
        return pred, h, c

# ===================== EVALUATION =====================

def evaluate_shared():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=Config.SPM_PATH)

    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    # Initialize model
    encoder = Encoder().to(device)
    shared_attn = SharedAttention().to(device)
    decoders = nn.ModuleDict({
        l: TribalDecoder(shared_attn).to(device)
        for l in Config.LANGS
    })

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {Config.MODEL_PATH}")

    state_dict = torch.load(Config.MODEL_PATH, map_location=device)

    # --- Load encoder ---
    enc_state = {
        k.replace("module.encoder.", "").replace("encoder.", ""): v
        for k, v in state_dict.items()
        if "encoder." in k
    }
    encoder.load_state_dict(enc_state)

    # --- Load decoders ---
    for l in Config.LANGS:
        dec_state = {
            k.replace(f"module.decoders.{l}.", "").replace(f"decoders.{l}.", ""): v
            for k, v in state_dict.items()
            if f"decoders.{l}." in k
        }
        decoders[l].load_state_dict(dec_state)

    encoder.eval()
    for d in decoders.values():
        d.eval()

    # ===================== PER LANGUAGE =====================

    for l_name in Config.LANGS:
        csv_path = os.path.join(Config.DATASET_DIR, Config.CSV_FILES[l_name])
        if not os.path.exists(csv_path):
            print(f"Skipping {l_name}: CSV not found")
            continue

        df = pd.read_csv(csv_path)
        limit = Config.EVAL_LIMITS.get(l_name)
        if limit is not None:
            df = df.head(limit)
            print(f"\nEvaluating {l_name.upper()} (limit={limit})")
        else:
            print(f"\nEvaluating {l_name.upper()} (full set)")

        results = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Inference [{l_name}]"):
            src_txt = str(row.iloc[0])
            tgt_txt = str(row.iloc[1])

            src_ids = sp.encode(src_txt, out_type=int, add_bos=True, add_eos=True)
            src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(device)

            with torch.no_grad():
                enc_outs, (h, c) = encoder(src_tensor)
                preds = [Config.BOS_ID]

                for _ in range(Config.MAX_LEN):
                    inp = torch.LongTensor([preds[-1]]).to(device)
                    out, h, c = decoders[l_name](inp, h, c, enc_outs)
                    nxt = out.argmax(1).item()
                    preds.append(nxt)
                    if nxt == Config.EOS_ID:
                        break

            pred_txt = sp.decode(preds)

            bleu_score = bleu.compute(
                predictions=[pred_txt],
                references=[[tgt_txt]],
                smooth_method="exp"
            )["score"]

            chrf_score = chrf.compute(
                predictions=[pred_txt],
                references=[[tgt_txt]]
            )["score"]

            results.append([src_txt, tgt_txt, pred_txt, bleu_score, chrf_score])

        out_df = pd.DataFrame(
            results,
            columns=["Hindi", f"Target_{l_name}", f"Predicted_{l_name}", "BLEU", "ChrF++"]
        )

        out_path = os.path.join(
            os.path.dirname(Config.MODEL_PATH),
            f"shared_eval_{l_name}.csv"
        )
        out_df.to_csv(out_path, index=False)

        print(f"Saved: {out_path}")
        print(f"Mean BLEU: {out_df['BLEU'].mean():.2f} | Mean ChrF++: {out_df['ChrF++'].mean():.2f}")

if __name__ == "__main__":
    evaluate_shared()
