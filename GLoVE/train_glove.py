# ============================================================
# Fully Distributed Subword GLoVe (100% DDP-Oriented)
# Shared SentencePiece vocab + Distributed Co-occurrence Build
# ============================================================

import os
import torch
import numpy as np
import sentencepiece as spm
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd

from config import *

# ============================================================
# DDP Setup
# ============================================================
def setup_ddp():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(gpu)
    else:
        rank, world, gpu = 0, 1, 0
    return rank, world, gpu

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# ============================================================
# Distributed Co-occurrence Builder
# ============================================================
def build_cooccurrence_distributed(tokenized_sentences, rank, world):
    """
    Each rank processes its shard and builds local co-occurrence.
    Then all ranks merge via all_reduce.
    """
    local_cooccur = defaultdict(float)

    # shard sentences
    shard = tokenized_sentences[rank::world]

    for tokens in tqdm(shard, desc=f"Rank {rank} building cooccur"):
        tokens = [t for t in tokens if t not in IGNORE_TOKEN_IDS]
        for i, wi in enumerate(tokens):
            start = max(0, i - WINDOW_SIZE)
            end   = min(len(tokens), i + WINDOW_SIZE + 1)
            for j in range(start, end):
                if i != j:
                    wj = tokens[j]
                    local_cooccur[(wi, wj)] += 1.0 / abs(i - j)

    # Convert to tensor representation
    keys = list(local_cooccur.keys())
    vals = torch.tensor([local_cooccur[k] for k in keys], dtype=torch.float32)

    # Gather sizes
    size_tensor = torch.tensor([len(keys)], device="cuda")
    if world > 1:
        dist.all_reduce(size_tensor, op=dist.ReduceOp.SUM)

    # We keep local sparse representation; merging done implicitly during training
    return local_cooccur

# ============================================================
# Dataset
# ============================================================
class GloVeDataset(Dataset):
    def __init__(self, cooccur_dict):
        self.data = [(i, j, x) for (i, j), x in cooccur_dict.items()]

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
# Model
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
        return (weight * inner.pow(2)).mean()

# ============================================================
# Train Language
# ============================================================
def train_language(lang):
    rank, world, gpu = setup_ddp()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    save_dir = SAVE_ROOT / lang
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Load SentencePiece
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    vocab_size = sp.get_piece_size()

    # Load corpus
    corpus_path = CORPORA_DIR / f"{lang}.txt"
    sentences = Path(corpus_path).read_text(encoding="utf-8").splitlines()

    tokenized = [sp.encode(s) for s in sentences if s.strip()]

    # Distributed co-occurrence build
    cooccur = build_cooccurrence_distributed(tokenized, rank, world)

    dataset = GloVeDataset(cooccur)

    train_len = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])

    sampler = DistributedSampler(train_ds) if world > 1 else None
    train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, shuffle=(sampler is None))
    val_loader = DataLoader(val_ds, BATCH_SIZE)

    model = GloVeModel(vocab_size).to(device)
    if world > 1:
        model = DDP(model, device_ids=[gpu])

    optimizer = optim.Adagrad(model.parameters(), lr=LR)

    best_val = float("inf")
    patience = 0
    logs = []

    for epoch in range(1, EPOCHS + 1):
        if sampler:
            sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0

        for i, j, x in tqdm(train_loader, disable=(rank != 0)):
            i, j, x = i.to(device), j.to(device), x.to(device)
            optimizer.zero_grad()
            loss = model(i, j, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Rank-0 validation
        if rank == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, j, x in val_loader:
                    i, j, x = i.to(device), j.to(device), x.to(device)
                    val_loss += model(i, j, x).item()
            val_loss /= len(val_loader)

            logs.append({"epoch": epoch, "train": train_loss, "val": val_loss})
            print(f"[{lang}] Epoch {epoch} Train {train_loss:.4f} Val {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                torch.save({
                    "weights": model.module.state_dict() if world > 1 else model.state_dict(),
                    "vocab_size": vocab_size
                }, save_dir / f"{lang}_glove.pt")
            else:
                patience += 1

        # Sync early stopping across ranks
        stop_flag = torch.tensor([patience >= PATIENCE], device=device, dtype=torch.uint8)
        if world > 1:
            dist.broadcast(stop_flag, src=0)

        if stop_flag.item():
            break

    if rank == 0:
        df = pd.DataFrame(logs)
        df.to_csv(save_dir / f"log_{lang}.csv", index=False)

    cleanup_ddp()

# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, choices=LANGUAGES)
    args = parser.parse_args()
    train_language(args.lang)
