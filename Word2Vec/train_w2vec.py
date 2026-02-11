# ============================================================
# Multilingual SentencePiece Word2Vec (SGNS)
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
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# PATHS & CONFIG
# ============================================================
BASE_DIR = Path("/home/kshhorizon/data/final-climb-shashwat-do-not-delete")
SAVE_ROOT = BASE_DIR / "Word2Vec"

LANG_CONFIG = {
    "Bhili":   {"file": "Hin_Bhi_Mar_Guj.csv", "col": 1},
    "Santali": {"file": "San_Hin_Eng.csv",     "col": 2},
    "Mundari": {"file": "Hin_Mun.csv",         "col": 1},
    "Gondi":   {"file": "Hin_Gon.csv",         "col": 1},
    "Kui":     {"file": "Hin_Kui.csv",         "col": 1},
}

# Hyperparameters
EMBED_DIM   = 50
WINDOW_SIZE = 5
NEG_SAMPLES = 20
BATCH_SIZE  = 8192
EPOCHS      = 10
MIN_COUNT   = 2
SUBSAMPLE_T = 1e-5
PATIENCE    = 5

# ============================================================
# DDP SETUP
# ============================================================
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(gpu)
    else:
        rank, world_size, gpu = 0, 1, 0
    return rank, world_size, gpu

# ============================================================
# DATASET
# ============================================================
class Word2VecDataset(Dataset):
    def __init__(self, sentences):
        self.window = WINDOW_SIZE

        # Count tokens (SentencePiece tokens already space-separated)
        word_counts = Counter()
        for s in sentences:
            word_counts.update(str(s).split())

        self.word_counts = word_counts
        self.total_count = sum(word_counts.values())

        # Build vocab
        self.id_to_word = [w for w, c in word_counts.items() if c >= MIN_COUNT] + ["<UNK>"]
        self.vocab = {w: i for i, w in enumerate(self.id_to_word)}

        # Negative sampling distribution
        counts = np.array([word_counts.get(w, 1) for w in self.id_to_word], dtype=np.float64)
        probs = np.power(counts, 0.75)
        self.noise_dist = torch.tensor(probs / probs.sum(), dtype=torch.float)

        # Build skip-gram pairs with subsampling
        self.data = []
        for s in sentences:
            tokens = []
            for w in str(s).split():
                if w in self.vocab:
                    f = word_counts[w] / self.total_count
                    p_keep = (np.sqrt(f / SUBSAMPLE_T) + 1) * (SUBSAMPLE_T / f)
                    if np.random.rand() < p_keep:
                        tokens.append(self.vocab[w])
                else:
                    tokens.append(self.vocab["<UNK>"])

            for i, target in enumerate(tokens):
                for j in range(max(0, i - self.window), min(len(tokens), i + self.window + 1)):
                    if i != j:
                        self.data.append((target, tokens[j]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t, c = self.data[idx]
        neg = torch.multinomial(self.noise_dist, NEG_SAMPLES, replacement=True)
        return torch.tensor(t), torch.tensor(c), neg

# ============================================================
# MODEL
# ============================================================
class SGNS(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.t = nn.Embedding(vocab_size, EMBED_DIM)
        self.c = nn.Embedding(vocab_size, EMBED_DIM)

        init = 0.5 / EMBED_DIM
        self.t.weight.data.uniform_(-init, init)
        self.c.weight.data.uniform_(-init, init)

    def forward(self, t, c, n):
        vt = self.t(t)
        vc = self.c(c)
        vn = self.c(n)

        pos_loss = -torch.log(
            torch.sigmoid((vt * vc).sum(1)) + 1e-10
        ).mean()

        neg_loss = -torch.log(
            torch.sigmoid(-torch.bmm(vn, vt.unsqueeze(2)).squeeze()) + 1e-10
        ).sum(1).mean()

        return pos_loss + neg_loss

# ============================================================
# TRAIN ONE LANGUAGE
# ============================================================
def train_language(lang):
    rank, world, gpu = setup_ddp()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    cfg = LANG_CONFIG[lang]
    df = pd.read_csv(BASE_DIR / "datasets" / cfg["file"])[:17500]
    sentences = df.iloc[:, cfg["col"]].astype(str).tolist()

    dataset = Word2VecDataset(sentences)
    train_len = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])

    sampler = DistributedSampler(train_ds) if world > 1 else None
    train_loader = DataLoader(
        train_ds, BATCH_SIZE, sampler=sampler, shuffle=(sampler is None)
    )
    val_loader = DataLoader(val_ds, BATCH_SIZE)

    model = SGNS(len(dataset.vocab)).to(device)
    if world > 1:
        model = DDP(model, device_ids=[gpu])

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    lang_dir = SAVE_ROOT / lang
    if rank == 0:
        lang_dir.mkdir(parents=True, exist_ok=True)
        log_rows = []

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

        for step, (t, c, n) in enumerate(pbar, start=1):
            t, c, n = t.to(device), c.to(device), n.to(device)
            optimizer.zero_grad()
            loss = model(t, c, n)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(train_loss=f"{train_loss / step:.4f}")

        train_loss /= len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for t, c, n in val_loader:
                t, c, n = t.to(device), c.to(device), n.to(device)
                val_loss += model(t, c, n).item()
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
                    lang_dir / f"{lang.lower()}_weights.pt",
                )
            else:
                patience += 1
                if patience >= PATIENCE:
                    break

    # ---------------- SAVE LOGS + PLOT ----------------
    if rank == 0:
        log_df = pd.DataFrame(log_rows)
        log_df.to_csv(lang_dir / f"train_log_{lang.lower()}_w2vec.csv", index=False)

        plt.figure(figsize=(8, 5))
        plt.plot(log_df["epoch"], log_df["train_loss"], label="Train")
        plt.plot(log_df["epoch"], log_df["val_loss"], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{lang} Word2Vec Training")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(lang_dir / f"loss_plot_{lang.lower()}_w2vec.png")
        plt.close()

    if world > 1:
        dist.destroy_process_group()

# ============================================================
# ENTRY POINT (MULTILINGUAL)
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, choices=LANG_CONFIG.keys())
    args = parser.parse_args()

    train_language(args.lang)

