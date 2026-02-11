# ============================================================
# Multilingual GLoVe Training
# With live tqdm train/val loss + logs + plots
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# PATHS & CONFIG
# ============================================================
BASE_DIR = Path("/home/shashwat1/final-climb-shashwat-do-not-delete")
SAVE_ROOT = BASE_DIR / "GLoVE"

LANG_CONFIG = {
    "Bhili":   {"file": "Hin_Bhi_Mar_Guj.csv", "col": 1},
    "Garo":    {"file": "Eng_Garo.csv",        "col": 1},
    "Gondi":   {"file": "Hin_Gon.csv",         "col": 1},
    "Kui":     {"file": "Hin_Kui.csv",         "col": 1},
    "Mundari": {"file": "Hin_Mun.csv",         "col": 1},
    "Santali": {"file": "San_Hin_Eng.csv",         "col": 2},
}

# Hyperparameters
EMBED_DIM   = 300
WINDOW_SIZE = 8
X_MAX       = 100
ALPHA       = 0.75
BATCH_SIZE  = 4096
EPOCHS      = 10
MIN_COUNT   = 2
PATIENCE    = 5

# ============================================================
# DDP SETUP
# ============================================================
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(gpu)
    else:
        rank, world, gpu = 0, 1, 0
    return rank, world, gpu

# ============================================================
# DATASET
# ============================================================
class GloVeDataset(Dataset):
    def __init__(self, sentences, lang_dir):
        # ---------------- Vocab ----------------
        word_counts = Counter()
        for s in tqdm(sentences, desc="Building vocab"):
            word_counts.update(str(s).split())

        self.id_to_word = [w for w, c in word_counts.items() if c >= MIN_COUNT]
        self.id_to_word.append("<UNK>")
        self.vocab = {w: i for i, w in enumerate(self.id_to_word)}
        self.vocab_size = len(self.vocab)

        # ---------------- Co-occurrence ----------------
        cooccur = defaultdict(float)
        for s in tqdm(sentences, desc="Building co-occurrence"):
            tokens = [self.vocab.get(w, self.vocab["<UNK>"]) for w in str(s).split()]
            for i, wi in enumerate(tokens):
                for j in range(
                    max(0, i - WINDOW_SIZE),
                    min(len(tokens), i + WINDOW_SIZE + 1),
                ):
                    if i != j:
                        cooccur[(wi, tokens[j])] += 1.0 / abs(i - j)

        self.data = [(i, j, x) for (i, j), x in cooccur.items()]

        if not dist.is_initialized() or dist.get_rank() == 0:
            lang_dir.mkdir(parents=True, exist_ok=True)
            torch.save(dict(cooccur), lang_dir / "cooccurrence_matrix.pt")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        i, j, x = self.data[idx]
        return (
            torch.tensor(i, dtype=torch.long),
            torch.tensor(j, dtype=torch.long),
            torch.tensor(x, dtype=torch.float),
        )

# ============================================================
# MODEL
# ============================================================
class GloVeModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.wi = nn.Embedding(vocab_size, EMBED_DIM)
        self.wj = nn.Embedding(vocab_size, EMBED_DIM)
        self.bi = nn.Embedding(vocab_size, 1)
        self.bj = nn.Embedding(vocab_size, 1)

        init = 0.5 / EMBED_DIM
        for p in self.parameters():
            nn.init.uniform_(p, -init, init)

    def forward(self, i, j, x):
        wi = self.wi(i)
        wj = self.wj(j)
        bi = self.bi(i).squeeze()
        bj = self.bj(j).squeeze()

        weight = torch.pow(torch.clamp(x / X_MAX, max=1.0), ALPHA)
        inner = torch.sum(wi * wj, dim=1) + bi + bj - torch.log(x)
        loss = weight * inner.pow(2)
        return loss.mean()

# ============================================================
# TRAIN ONE LANGUAGE
# ============================================================
def train_language(lang):
    rank, world, gpu = setup_ddp()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    cfg = LANG_CONFIG[lang]
    df = pd.read_csv(BASE_DIR / "datasets" / cfg["file"])[:7500]
    sentences = df.iloc[:, cfg["col"]].astype(str).tolist()

    lang_dir = SAVE_ROOT / lang
    if rank == 0:
        lang_dir.mkdir(parents=True, exist_ok=True)
        log_rows = []

    dataset = GloVeDataset(sentences, lang_dir)
    train_len = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])

    sampler = DistributedSampler(train_ds) if world > 1 else None
    train_loader = DataLoader(
        train_ds, BATCH_SIZE, sampler=sampler, shuffle=(sampler is None)
    )
    val_loader = DataLoader(val_ds, BATCH_SIZE)

    model = GloVeModel(dataset.vocab_size).to(device)
    if world > 1:
        model = DDP(model, device_ids=[gpu])

    optimizer = optim.Adagrad(model.parameters(), lr=0.05)

    best_val = float("inf")
    patience = 0

    for epoch in range(1, EPOCHS + 1):
        if sampler:
            sampler.set_epoch(epoch)

        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"[{lang}] Epoch {epoch}/{EPOCHS}",
            disable=(rank != 0),
        )

        for step, (i, j, x) in enumerate(pbar, start=1):
            i, j, x = i.to(device), j.to(device), x.to(device)
            optimizer.zero_grad()
            loss = model(i, j, x)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(train_loss=f"{train_loss / step:.4f}")

        train_loss /= len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, j, x in val_loader:
                i, j, x = i.to(device), j.to(device), x.to(device)
                val_loss += model(i, j, x).item()
        val_loss /= len(val_loader)

        # ---------------- LOGGING ----------------
        if rank == 0:
            log_rows.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )

            tqdm.write(
                f"[{lang}] Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                torch.save(
                    {
                        "vocab": dataset.vocab,
                        "id_to_word": dataset.id_to_word,
                        "weights": model.module.state_dict()
                        if world > 1
                        else model.state_dict(),
                    },
                    lang_dir / f"{lang.lower()}_glove.pt",
                )
            else:
                patience += 1
                if patience >= PATIENCE:
                    break

    # ---------------- SAVE LOGS + PLOT ----------------
    if rank == 0:
        log_df = pd.DataFrame(log_rows)
        log_df.to_csv(lang_dir / f"train_log_{lang.lower()}_glove.csv", index=False)

        plt.figure(figsize=(8, 5))
        plt.plot(log_df["epoch"], log_df["train_loss"], label="Train")
        plt.plot(log_df["epoch"], log_df["val_loss"], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{lang} GloVe Training")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(lang_dir / f"loss_plot_{lang.lower()}_glove.png")
        plt.close()

    if world > 1:
        dist.destroy_process_group()

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, choices=LANG_CONFIG.keys())
    args = parser.parse_args()

    train_language(args.lang)
