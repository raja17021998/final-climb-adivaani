import os
import random
import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, random_split
from tqdm import tqdm
import pandas as pd
from safetensors.torch import save_model
import sentencepiece as spm
from pathlib import Path
import matplotlib.pyplot as plt

from transformers import get_linear_schedule_with_warmup

from bert_config import BertConfig
from model_bert import BertModel
from mlm_head import MLMHead
from dataset_mlm import MLMDataset

BASE_DIR = "/home/jovyan/final-climb-shashwat-do-not-delete"
SAVE_ROOT = f"{BASE_DIR}/BERT"
TOKENIZER_PATH = f"{BASE_DIR}/tokenization/joint_spm.model"


# =========================
# DDP SETUP
# =========================
def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.barrier()
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def load_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    return sp


# =========================
# TRAIN
# =========================
def train(lang, token_ids, epochs, patience, min_delta, batch_size):

    rank, world = setup_ddp()
    device = torch.device("cuda")

    # ---- deterministic seed across ranks ----
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    sp = load_tokenizer()
    pad_id = sp.pad_id()
    cls_id = sp.bos_id()
    sep_id = sp.eos_id()
    mask_id = sp.piece_to_id("<mask>")

    cfg = BertConfig(vocab_size=sp.get_piece_size())

    out_dir = f"{SAVE_ROOT}/{lang}"
    logs_dir = f"{out_dir}/LOGS"
    loss_dir = f"{out_dir}/LOSS"
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(loss_dir, exist_ok=True)

    # =========================
    # Dataset (deterministic split)
    # =========================
    dataset_full = MLMDataset(token_ids, cfg, cls_id, sep_id, mask_id, pad_id)

    val_size = int(0.1 * len(dataset_full))
    train_size = len(dataset_full) - val_size

    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset_full, [train_size, val_size], generator=g)

    train_sampler = DistributedSampler(train_ds) if world > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        sampler=val_sampler,
        shuffle=False,
        pin_memory=True
    )



    train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False
    )


    # =========================
    # Model
    # =========================
    model = nn.ModuleDict({
        "bert": BertModel(cfg),
        "mlm": MLMHead(cfg),
    }).to(device)

    # weight tying
    model["mlm"].net[-1].weight = model["bert"].emb.weight

    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            find_unused_parameters=False
        )

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps
    )

    logs = []
    best_val_loss = float("inf")
    patience_counter = 0

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(1, epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, disable=(rank != 0))

        for ids, labels in pbar:
            ids = ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pad_mask = (ids == pad_id)

            with torch.amp.autocast("cuda"):
                hidden = model["bert"](ids, pad_mask)
                logits = model["mlm"](hidden)
                loss = loss_fn(logits.view(-1, cfg.vocab_size), labels.view(-1))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        # ---- reduce train loss across GPUs ----
        train_loss = torch.tensor(train_loss, device=device)
        if world > 1:
            dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        train_loss = train_loss.item() / (len(train_loader) * world)

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for ids, labels in val_loader:
                ids = ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                pad_mask = (ids == pad_id)

                hidden = model["bert"](ids, pad_mask)
                logits = model["mlm"](hidden)
                loss = loss_fn(logits.view(-1, cfg.vocab_size), labels.view(-1))
                val_loss += loss.item()

        # ---- reduce val loss across GPUs ----
        val_loss = torch.tensor(val_loss, device=device)
        if world > 1:
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss = val_loss.item() / (len(val_loader) * world)

        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)


        # =========================
        # EARLY STOPPING (synced)
        # =========================
        improved = (best_val_loss - val_loss) > min_delta
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        stop_tensor = torch.tensor(0, device=device)
        if patience_counter >= patience:
            stop_tensor.fill_(1)

        if world > 1:
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.SUM)

        if rank == 0:
            print(
                f"Epoch {epoch} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}"
            )
            logs.append([epoch, train_loss, val_loss, train_ppl, val_ppl])

        if stop_tensor.item() > 0:
            if rank == 0:
                print(f"\nEarly stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
            break

    # =========================
    # SAVE (rank 0 only)
    # =========================
    if rank == 0:
        df = pd.DataFrame(logs, columns=["epoch", "train_loss", "val_loss", "train_ppl", "val_ppl"])
        df.to_csv(f"{logs_dir}/metrics.csv", index=False)

        # ---- Loss plot ----
        plt.figure()
        plt.plot(df.epoch, df.train_loss, label="Train")
        plt.plot(df.epoch, df.val_loss, label="Val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"{loss_dir}/loss_curve.png")
        plt.close()

        # ---- Perplexity plot (clip extreme values only for visualization) ----
        plt.figure()

        ppl_cap = 1000  # visualization cap
        train_ppl_plot = [min(p, ppl_cap) for p in df.train_ppl]
        val_ppl_plot = [min(p, ppl_cap) for p in df.val_ppl]

        plt.plot(df.epoch, train_ppl_plot, label="Train")
        plt.plot(df.epoch, val_ppl_plot, label="Val")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Perplexity (clipped)")

        # show "..." if clipping occurred
        if max(df.train_ppl.max(), df.val_ppl.max()) > ppl_cap:
            plt.text(
                0.98, 0.95, "...",
                transform=plt.gca().transAxes,
                ha="right",
                va="top",
                fontsize=16
            )

        plt.savefig(f"{loss_dir}/perplexity_curve.png")
        plt.close()

        save_model(model.module if world > 1 else model, f"{out_dir}/model.safetensors")

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()



# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--max_lines", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    args = parser.parse_args()

    sp = load_tokenizer()

    corpus_path = f"{BASE_DIR}/tokenization/corpora/{args.lang.lower()}.txt"
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Missing corpus file: {corpus_path}")

    sentences = Path(corpus_path).read_text().splitlines()
    if args.max_lines:
        sentences = sentences[:args.max_lines]

    token_ids = [sp.encode(s) for s in sentences if s.strip()]

    train(
    args.lang,
    token_ids,
    args.epochs,
    args.early_stopping_patience,
    args.early_stopping_min_delta,
    args.batch_size
)

