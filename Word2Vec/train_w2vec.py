# ============================================================
# Multilingual SentencePiece Word2Vec (SGNS)
# Fully Config-Driven | SPD Auto Scaling | DDP Compatible
# Optimized Negative Sampling + SentencePiece Integration
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import sentencepiece as spm

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

import config


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
# DATASET (NO NEGATIVE SAMPLING HERE)
# ============================================================

class Word2VecDataset(Dataset):
    def __init__(self, sentences, window_size, min_count, subsample_t):

        self.window = window_size
        self.subsample_t = subsample_t

        word_counts = Counter()
        for s in sentences:
            word_counts.update(s.split())

        self.word_counts = word_counts
        self.total_count = sum(word_counts.values())

        # Build vocab
        self.id_to_word = [w for w, c in word_counts.items() if c >= min_count]
        self.id_to_word.append("<UNK>")
        self.vocab = {w: i for i, w in enumerate(self.id_to_word)}

        # Negative sampling distribution
        counts = np.array(
            [word_counts.get(w, 1) for w in self.id_to_word],
            dtype=np.float64,
        )
        probs = np.power(counts, 0.75)
        self.noise_dist = torch.tensor(probs / probs.sum(), dtype=torch.float)

        # Build skip-gram pairs
        self.data = []
        for s in sentences:
            tokens = []

            for w in s.split():
                if w in self.vocab:
                    f = word_counts[w] / self.total_count
                    p_keep = (np.sqrt(f / self.subsample_t) + 1) * (
                        self.subsample_t / f
                    )
                    if np.random.rand() < p_keep:
                        tokens.append(self.vocab[w])
                else:
                    tokens.append(self.vocab["<UNK>"])

            for i, target in enumerate(tokens):
                for j in range(
                    max(0, i - self.window),
                    min(len(tokens), i + self.window + 1),
                ):
                    if i != j:
                        self.data.append((target, tokens[j]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t, c = self.data[idx]
        return torch.tensor(t), torch.tensor(c)


# ============================================================
# MODEL
# ============================================================

class SGNS(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        self.t = nn.Embedding(vocab_size, embed_dim)
        self.c = nn.Embedding(vocab_size, embed_dim)

        init = 0.5 / embed_dim
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
            torch.sigmoid(
                -torch.bmm(vn, vt.unsqueeze(2)).squeeze()
            ) + 1e-10
        ).sum(1).mean()

        return pos_loss + neg_loss


# ============================================================
# TRAIN ONE LANGUAGE
# ============================================================

def train_language(lang):

    rank, world, gpu = setup_ddp()
    device = torch.device(
        f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    )

    # ---------------- CONFIG ----------------
    cfg = config.TRIBAL_LANG_CONFIG[lang]
    spd = config.compute_spd(lang)
    params = config.get_scale_params(spd)

    EMBED_DIM   = config.EMBED_DIM
    WINDOW_SIZE = params["WINDOW_SIZE"]
    NEG_SAMPLES = params["NEG_SAMPLES"]
    MIN_COUNT   = params["MIN_COUNT"]
    EPOCHS      = params["EPOCHS"]
    SUBSAMPLE_T = params["SUBSAMPLE_T"]

    BATCH_SIZE  = config.BATCH_SIZE
    LR          = config.LR
    PATIENCE    = config.PATIENCE

    if rank == 0:
        print(f"\n========== {lang} ==========")
        print(f"SPD: {spd}")
        print(f"Params: {params}")
        print(f"Embedding Dim: {EMBED_DIM}")
        print("============================\n")

    # ---------------- LOAD DATA ----------------
    df = pd.read_csv(config.BASE_DATA_DIR / cfg["file"])
    raw_sentences = df.iloc[:, cfg["tribal_col"]].astype(str).tolist()

    # ---------------- LOAD SENTENCEPIECE ----------------
    sp = spm.SentencePieceProcessor()
    sp.load("/home/jovyan/final-climb-shashwat-do-not-delete/tokenization/joint_spm.model")

    # Encode into subword space
    sentences = []
    for s in raw_sentences:
        pieces = sp.encode(s, out_type=str)
        if pieces:
            sentences.append(" ".join(pieces))

    # ---------------- DATASET ----------------
    dataset = Word2VecDataset(
        sentences,
        WINDOW_SIZE,
        MIN_COUNT,
        SUBSAMPLE_T,
    )

    noise_dist = dataset.noise_dist.to(device)

    train_len = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(
        dataset, [train_len, len(dataset) - train_len]
    )

    sampler = DistributedSampler(train_ds) if world > 1 else None

    train_loader = DataLoader(
        train_ds,
        BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
    )

    val_loader = DataLoader(val_ds, BATCH_SIZE)

    model = SGNS(len(dataset.vocab), EMBED_DIM).to(device)

    if world > 1:
        model = DDP(model, device_ids=[gpu])

    optimizer = optim.Adam(model.parameters(), lr=LR)

    lang_dir = config.SAVE_ROOT / lang
    if rank == 0:
        lang_dir.mkdir(parents=True, exist_ok=True)
        log_rows = []

    best_val = float("inf")
    patience_counter = 0

    # ============================================================
    # TRAIN LOOP
    # ============================================================

    for epoch in range(1, EPOCHS + 1):

        if sampler:
            sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"[{lang}] Epoch {epoch}/{EPOCHS}",
            disable=(rank != 0),
        )

        for step, (t, c) in enumerate(pbar, start=1):

            t, c = t.to(device), c.to(device)
            batch_size = t.size(0)

            neg = torch.multinomial(
                noise_dist,
                batch_size * NEG_SAMPLES,
                replacement=True,
            ).view(batch_size, NEG_SAMPLES)

            optimizer.zero_grad()
            loss = model(t, c, neg)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(
                train_loss=f"{train_loss / step:.4f}"
            )

        train_loss /= len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for t, c in val_loader:
                t, c = t.to(device), c.to(device)
                batch_size = t.size(0)

                neg = torch.multinomial(
                    noise_dist,
                    batch_size * NEG_SAMPLES,
                    replacement=True,
                ).view(batch_size, NEG_SAMPLES)

                val_loss += model(t, c, neg).item()

        val_loss /= len(val_loader)

        # ---------------- EARLY STOPPING ----------------
        if rank == 0:

            log_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

            tqdm.write(
                f"[{lang}] Epoch {epoch} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0

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
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("Early stopping triggered.")
                    break

    if world > 1:
        dist.destroy_process_group()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=config.TRIBAL_LANG_CONFIG.keys(),
    )

    args = parser.parse_args()

    train_language(args.lang)
