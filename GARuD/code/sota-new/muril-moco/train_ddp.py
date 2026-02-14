import os
import gc
import logging
import socket
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn

from transformers import AutoTokenizer, AutoModelForMaskedLM, get_linear_schedule_with_warmup
import pandas as pd

from config import *
from utils import *

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt / world_size


def load_data(path, max_rows=None):
    """Read CSV, clean, and return parallel Hindi/Bhili lists."""
    df = pd.read_csv(path)
    df = df.drop(columns=["Marathi", "Gujarati"])
    df = df.dropna(subset=["Hindi", "Bhili"])
    df = df[df["Hindi"].str.strip().astype(bool)].reset_index(drop=True)
    if max_rows:
        df = df.head(max_rows)
    return df["Hindi"].tolist(), df["Bhili"].tolist()


def build_loaders(hi, bh, student_tokenizer, teacher_tokenizer, device, world_size, rank, shuffle):
    ds = ParallelDataset(hi, bh)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle) 
    collate = DistillCollatorCustom(
        teacher_tok=teacher_tokenizer,           # not used for teacher
        student_tok=student_tokenizer,
        max_len=MAX_SEQ_LEN,
        mlm_prob=MLM_PROB,
        device=device,
    )
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
    )


def run_epoch(model, teacher, loader, optimizer, scheduler, scaler, loss_fn, is_train, device):
    model.train(is_train)
    total_ce, total_distill, total_acc, n = 0.0, 0.0, 0.0, 0

    for batch in tqdm(loader, desc="Train" if is_train else "Val", leave=False):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with autocast():
            # ---------- Student ----------
            out = model(
                input_ids=batch["bh_input_ids"],
                attention_mask=batch["bh_attention_mask"],
                output_hidden_states=True,
            )
            logits = out.logits
            hidden = out.hidden_states[-1]
            stud_emb = hidden[:, 0, :].clone()

            ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.size(-1)), batch["bh_labels"].view(-1)
            )

            # ---------- Teacher ----------
            with torch.no_grad():
                teach_out = teacher(
                    input_ids=batch["hi_input_ids"],
                    attention_mask=batch["hi_attention_mask"],
                    output_hidden_states=True,
                )
                teach_emb = teach_out.hidden_states[-1][:, 0, :].clone()

            distill_loss = loss_fn(stud_emb, teach_emb)

            # ---------- Combine ----------
            loss = ce_loss + distill_loss

        # accuracy
        preds = logits.argmax(-1)
        mask = batch["bh_labels"] != -100
        acc = ((preds == batch["bh_labels"]) & mask).sum().item() / mask.sum().item()


        if is_train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        total_ce += ce_loss.item()
        total_distill += distill_loss.item()
        total_acc += acc
        n += 1

    return total_ce / n, total_distill / n, total_acc / n


def main_worker(rank, world_size):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(LOGGING_DIR, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(LOGGING_DIR, f"train_{datetime.now():%Y%m%d_%H%M%S}.log"),
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s"
        )

    cudnn.benchmark = True

    # ---------- Data ----------
    hindi_train, bhili_train = load_data(
        "../../../datasets/train_90p.csv", max_rows=None
    )
    hindi_val, bhili_val = load_data(
        "../../../datasets/val_10p.csv", max_rows=None
    )

    # ---------- Models ----------
    teacher_tok = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "hindi-bert-v2"), local_files_only=True)
    student_tok = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "muril-base-cased"), local_files_only=True)

    teacher = (
        AutoModelForMaskedLM.from_pretrained(os.path.join(MODEL_DIR, "hindi-bert-v2"), local_files_only=True)
        .eval()
        .to(device)
    )
    for p in teacher.parameters():
        p.requires_grad_(False)

    if not os.path.exists("checkpoint.pth"):
        student = (
        AutoModelForMaskedLM.from_pretrained(os.path.join(MODEL_DIR, "muril-base-cased"), local_files_only=True)
        .to(device)
        )
    else:
        print("LOADING FROM OUTPUT_DIR")
        student = (
        AutoModelForMaskedLM.from_pretrained(OUTPUT_DIR, local_files_only=True)
        .to(device)
        )
    student = DDP(student, device_ids = [rank])
    #student.gradient_checkpointing_enable()

    # ---------- DataLoaders ----------
    train_loader = build_loaders(hindi_train, bhili_train, student_tok, teacher_tok, device, world_size, rank, shuffle=True)
    val_loader   = build_loaders(hindi_val,   bhili_val,   student_tok, teacher_tok, device, world_size, rank, shuffle=False)

    # ---------- Loss, Optimizer, Scheduler ----------
    contrastive_loss_fn = DebiasedNTXent(
        feature_dim=768, queue_size=4096, temperature=0.1, device=device
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )
    scaler = GradScaler()

    # ---------- Training ----------
    best_val_acc = float("-inf")
    start_epoch = 1

    if os.path.exists("checkpoint.pth"):
        checkpoint = torch.load('checkpoint.pth', map_location=device)

        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        best_val_acc = checkpoint["best_val_acc"]

    for epoch in range(start_epoch, EPOCHS + 1):
        tr_ce, tr_distill, tr_acc = run_epoch(
            student, teacher, train_loader, optimizer, scheduler, scaler, contrastive_loss_fn, True, device
        )
        val_ce, val_distill, val_acc = run_epoch(
            student, teacher, val_loader, None, None, None, contrastive_loss_fn, False, device
        )

        # tqdm.write(
        #     f"[Epoch {epoch:03d}] "
        #     f"Train MLM {tr_ce:.4f}  Distill {tr_distill:.4f}  Acc {tr_acc:.4f} | "
        #     f"Val   MLM {val_ce:.4f}  Distill {val_distill:.4f}  Acc {val_acc:.4f}"
        # )
        train_loss = reduce_tensor(torch.tensor(tr_ce + tr_distill, device=device), world_size).item()
        train_acc = reduce_tensor(torch.tensor(tr_acc, device=device), world_size).item()
        train_distill = reduce_tensor(torch.tensor(tr_distill, device=device), world_size).item()
        train_ce = reduce_tensor(torch.tensor(tr_ce, device=device), world_size).item()

        val_loss = reduce_tensor(torch.tensor(val_ce + val_distill, device=device), world_size).item()
        val_acc = reduce_tensor(torch.tensor(val_acc, device=device), world_size).item()
        val_distill = reduce_tensor(torch.tensor(val_distill, device=device), world_size).item()
        val_ce = reduce_tensor(torch.tensor(val_ce, device=device), world_size).item()

        # Save best checkpoint
        if rank == 0:
            logging.info(f"[Epoch {epoch}] Train: MLM={train_ce:.4f} Contrastive={train_distill:.4f} Total={train_loss:.4f} Acc={train_acc:.4f} | Val: MLM={val_ce:.4f} Contrastive={val_distill:.4f} Total={val_loss:.4f} Acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint = {
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # For mixed precision
                'best_val_acc': best_val_acc
                }

                torch.save(checkpoint, 'checkpoint.pth')
                student.module.save_pretrained(OUTPUT_DIR)
                student_tok.save_pretrained(OUTPUT_DIR)
                logging.info(f"Saved best model to {OUTPUT_DIR} at epoch {epoch}")

        gc.collect()
        torch.cuda.empty_cache()

    cleanup_ddp()


def main():
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("DDP requires at least 2 GPUs. Exiting.")
        return

    # 🔐 Set env vars before spawning processes
    master_addr = socket.gethostbyname(socket.gethostname())
    master_port = find_free_port()
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    print(f"Launching DDP with {world_size} GPUs")
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)            


if __name__ == "__main__":
    main()
