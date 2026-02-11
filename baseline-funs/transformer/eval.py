import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import pandas as pd
import sentencepiece as spm
import evaluate
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ===================== CONFIG =====================

class Config:
    ROOT_DIR = "/home/user/Desktop/Shashwat/final-climb/baseline-funs"
    DATASET_DIR = "/home/user/Desktop/Shashwat/final-climb/datasets"
    SPM_PATH = os.path.join(ROOT_DIR, "joint_spm.model")

    # 🔽 MUST MATCH TRAINING
    TGT_LANG = "mundari"
    CSV_FILE = "Hin_Mun.csv"
    MODEL_PATH = os.path.join(ROOT_DIR, "transformer", TGT_LANG.capitalize(), "best.pt")

    VOCAB_SIZE = 8000
    D_MODEL = 512
    N_HEADS = 8
    ENC_LAYERS = 6
    DEC_LAYERS = 6
    FF_DIM = 2048
    DROPOUT = 0.1

    # 🔽 Beam Search
    BEAM_WIDTH = 5
    MAX_LEN = 60

    BOS_ID = 1
    EOS_ID = 2
    PAD_ID = 0

    EVAL_LIMIT = 1000  # None for full CSV

# ===================== POSITIONAL ENCODING =====================

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x, start=0):
        return x + self.pe[start:start + x.size(0)]

# ===================== MODEL =====================

class TransformerNMT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(Config.VOCAB_SIZE, Config.D_MODEL)
        self.pe = SinusoidalPE(Config.D_MODEL)

        enc_layer = nn.TransformerEncoderLayer(
            Config.D_MODEL, Config.N_HEADS, Config.FF_DIM, Config.DROPOUT
        )
        dec_layer = nn.TransformerDecoderLayer(
            Config.D_MODEL, Config.N_HEADS, Config.FF_DIM, Config.DROPOUT
        )

        self.encoder = nn.TransformerEncoder(enc_layer, Config.ENC_LAYERS)
        self.decoder = nn.TransformerDecoder(dec_layer, Config.DEC_LAYERS)
        self.fc_out = nn.Linear(Config.D_MODEL, Config.VOCAB_SIZE)

    def encode(self, src):
        return self.encoder(self.pe(self.emb(src)))

    def decode_step(self, tgt, memory, step):
        tgt_emb = self.pe(self.emb(tgt), start=step)
        out = self.decoder(tgt_emb, memory)
        logits = self.fc_out(out[-1])
        return torch.log_softmax(logits, dim=-1)

# ===================== BEAM SEARCH =====================

@torch.no_grad()
def beam_search_decode(model, memory, beam_width):
    device = memory.device

    beams = [([Config.BOS_ID], 0.0)]  # (tokens, log_prob)

    for step in range(Config.MAX_LEN):
        new_beams = []

        for tokens, score in beams:
            if tokens[-1] == Config.EOS_ID:
                new_beams.append((tokens, score))
                continue

            tgt = torch.tensor(tokens, device=device).unsqueeze(1)
            log_probs = model.decode_step(tgt, memory, step)  # (1, vocab)

            topk_logp, topk_ids = torch.topk(log_probs, beam_width)

            topk_logp = topk_logp.squeeze(0)
            topk_ids = topk_ids.squeeze(0)

            for lp, idx in zip(topk_logp.tolist(), topk_ids.tolist()):
                new_beams.append((tokens + [idx], score + float(lp)))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        if all(b[0][-1] == Config.EOS_ID for b in beams):
            break

    return beams[0][0]


# ===================== EVALUATION =====================

def main():
    world_size = torch.cuda.device_count()
    rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    sp = spm.SentencePieceProcessor(model_file=Config.SPM_PATH)

    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    model = TransformerNMT().to(device)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    df = pd.read_csv(os.path.join(Config.DATASET_DIR, Config.CSV_FILE))
    if Config.EVAL_LIMIT:
        df = df.head(Config.EVAL_LIMIT)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), disable=(rank != 0)):
        src_txt = str(row.iloc[0])
        tgt_txt = str(row.iloc[1])

        src_ids = sp.encode(src_txt, out_type=int, add_bos=True, add_eos=True)
        src = torch.tensor(src_ids, device=device).unsqueeze(1)

        with torch.no_grad():
            memory = model.encode(src)
            pred_ids = beam_search_decode(model, memory, Config.BEAM_WIDTH)

        pred_txt = sp.decode(pred_ids)

        b = bleu.compute(
            predictions=[pred_txt],
            references=[[tgt_txt]],
            smooth_method="exp"
        )["score"]

        c = chrf.compute(
            predictions=[pred_txt],
            references=[[tgt_txt]]
        )["score"]

        results.append([src_txt, tgt_txt, pred_txt, b, c])

    # -------- SAVE --------
    if rank == 0:
        out_dir = os.path.dirname(Config.MODEL_PATH)
        out_path = os.path.join(out_dir, f"eval_beam{Config.BEAM_WIDTH}.csv")

        out_df = pd.DataFrame(
            results,
            columns=["Hindi", "Target", "Predicted", "BLEU", "ChrF++"]
        )
        out_df.to_csv(out_path, index=False)

        print(f"\nSaved evaluation to: {out_path}")
        print(f"Mean BLEU: {out_df['BLEU'].mean():.2f}")
        print(f"Mean ChrF++: {out_df['ChrF++'].mean():.2f}")

if __name__ == "__main__":
    main()
