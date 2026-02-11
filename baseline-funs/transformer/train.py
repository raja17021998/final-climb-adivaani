# /home/user/Desktop/Shashwat/final-climb/baseline-funs/transformer/train.py

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------

class Config:
    ROOT_DIR = "/home/user/Desktop/Shashwat/final-climb/baseline-funs"
    DATASET_DIR = "/home/user/Desktop/Shashwat/final-climb/datasets"
    SPM_PATH = os.path.join(ROOT_DIR, "joint_spm.model")

    # 🔽 CHANGE THIS PER RUN
    TGT_LANG = "Mundari"
    CSV_FILE = "Hin_Mun.csv"

    SAVE_DIR = os.path.join(
        ROOT_DIR, "transformer", TGT_LANG.capitalize()
    )

    VOCAB_SIZE = 8000
    D_MODEL = 512
    N_HEADS = 8
    ENC_LAYERS = 6
    DEC_LAYERS = 6
    FF_DIM = 2048
    DROPOUT = 0.1

    BATCH_SIZE = 16
    LR = 3e-4
    EPOCHS = 5
    VAL_SPLIT = 0.1

    PAD_ID = 0

os.makedirs(Config.SAVE_DIR, exist_ok=True)

# ---------------- DATA ----------------

class NMTDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.sp = spm.SentencePieceProcessor(model_file=Config.SPM_PATH)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = self.sp.encode(
            str(self.df.iloc[idx, 0]),
            out_type=int,
            add_bos=True,
            add_eos=True
        )
        tgt = self.sp.encode(
            str(self.df.iloc[idx, 1]),
            out_type=int,
            add_bos=True,
            add_eos=True
        )
        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    src, tgt = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, padding_value=Config.PAD_ID)
    tgt = nn.utils.rnn.pad_sequence(tgt, padding_value=Config.PAD_ID)
    return src, tgt

# ---------------- POSITIONAL ENCODING ----------------

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[:x.size(0)]

# ---------------- MODEL ----------------

class TransformerNMT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(Config.VOCAB_SIZE, Config.D_MODEL)
        self.pe = SinusoidalPE(Config.D_MODEL)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=Config.D_MODEL,
            nhead=Config.N_HEADS,
            dim_feedforward=Config.FF_DIM,
            dropout=Config.DROPOUT
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=Config.D_MODEL,
            nhead=Config.N_HEADS,
            dim_feedforward=Config.FF_DIM,
            dropout=Config.DROPOUT
        )

        self.encoder = nn.TransformerEncoder(enc_layer, Config.ENC_LAYERS)
        self.decoder = nn.TransformerDecoder(dec_layer, Config.DEC_LAYERS)
        self.fc_out = nn.Linear(Config.D_MODEL, Config.VOCAB_SIZE)

    def forward(self, src, tgt):
        src_emb = self.pe(self.emb(src))
        tgt_emb = self.pe(self.emb(tgt[:-1]))
        memory = self.encoder(src_emb)
        out = self.decoder(tgt_emb, memory)
        return self.fc_out(out)

# ---------------- TRAIN ----------------

def main():
    world_size = torch.cuda.device_count()
    rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Load & split data
    full_df = pd.read_csv(os.path.join(Config.DATASET_DIR, Config.CSV_FILE))
    full_df= full_df[:10000]
    val_size = int(len(full_df) * Config.VAL_SPLIT)
    train_df = full_df.iloc[:-val_size]
    val_df = full_df.iloc[-val_size:]

    train_ds = NMTDataset(train_df)
    val_ds = NMTDataset(val_df)

    train_sampler = DistributedSampler(train_ds) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = TransformerNMT().to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_ID)

    history = []

    for epoch in range(Config.EPOCHS):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        train_loss = 0
        for src, tgt in tqdm(train_loader, disable=(rank != 0)):
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt)
            loss = criterion(
                logits.reshape(-1, Config.VOCAB_SIZE),
                tgt[1:].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                logits = model(src, tgt)
                loss = criterion(
                    logits.reshape(-1, Config.VOCAB_SIZE),
                    tgt[1:].reshape(-1)
                )
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if rank == 0:
            print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            history.append([epoch, train_loss, val_loss])
            torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, "best.pt"))

    # -------- SAVE LOGS --------
    if rank == 0:
        df = pd.DataFrame(history, columns=["Epoch", "TrainLoss", "ValLoss"])
        df.to_csv(os.path.join(Config.SAVE_DIR, "losses.csv"), index=False)

        plt.figure(figsize=(8,5))
        plt.plot(df["TrainLoss"], label="Train")
        plt.plot(df["ValLoss"], label="Val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(Config.TGT_LANG.capitalize())
        plt.savefig(os.path.join(Config.SAVE_DIR, "losses.png"))

if __name__ == "__main__":
    main()
