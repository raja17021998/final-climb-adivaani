import os 
import gc
from tqdm import tqdm 
import logging
from datetime import datetime
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_linear_schedule_with_warmup

from config import *
from utils import * 


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    logging.basicConfig(
    filename=os.path.join(LOGGING_DIR, f"train_{datetime.now():%Y%m%d_%H%M%S}.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
    )

    cudnn.benchmark = True
    cudnn.deterministic = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CARD)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_CARD)

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        model = nn.DataParallel(model)
    model.to(DEVICE)
    model = torch.compile(model)
    
    train_df = pd.read_csv("/teamspace/studios/this_studio/train_90p.csv")
    val_df = pd.read_csv("/teamspace/studios/this_studio/val_10p.csv")
    train_df = train_df.drop(columns=["Marathi", "Gujarati"])
    train_df = train_df.dropna(subset=["Hindi", "Bhili"]).reset_index(drop=True)
    train_df[train_df["Hindi"].str.strip().astype(bool)].reset_index(drop=True)
    bhili_texts_train = train_df["Bhili"].tolist()[:50000]  # Limit to 10,000 samples for training

    val_df = val_df.drop(columns=["Marathi", "Gujarati"])
    val_df = val_df.dropna(subset=["Hindi", "Bhili"]).reset_index(drop=True)
    val_df = val_df[val_df["Hindi"].str.strip().astype(bool)].reset_index(drop=True)
    bhili_texts_val = val_df["Bhili"].tolist()[:5000]

    del train_df, val_df
    gc.collect()

    train_set = BhiliDataset(bhili_texts_train)
    val_set = BhiliDataset(bhili_texts_val)
    collate_fn = DistillCollatorCustom(
        student_tok=tokenizer,
        max_len=MAX_SEQ_LEN,
        mlm_prob=MLM_PROB,
        device=DEVICE
    )
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    # Optimizer and scheduler
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    num_training_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1*num_training_steps), num_training_steps=num_training_steps
    )
    scaler = GradScaler()

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, ce_tot, acc_tot = 0, 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            # Forward passes
            # Student (Bhili)
            for k, v in batch.items():
                batch[k] = v.to(DEVICE, non_blocking=True)
            
            inputs = {
                    "input_ids": batch["bh_input_ids"],
                    "attention_mask": batch["bh_attention_mask"]
                }
            with autocast():
                outputs = model(**inputs, output_hidden_states=True)    
                logits  = outputs.logits    # [B, T, V_student]

                mlm_loss = ce_loss_fn(
                    logits.view(-1, logits.size(-1)),
                    batch["bh_labels"].view(-1)
                )
            
        
            # Compute MLM accuracy
            predictions = logits.argmax(dim=-1)  # [batch, seq_len]
            masked_positions = batch["bh_labels"] != -100  # [batch, seq_len]
            correct = (predictions == batch["bh_labels"]) & masked_positions  # [batch, seq_len]
            num_correct = correct.sum().item()
            num_masked = masked_positions.sum().item()
            mlm_accuracy = num_correct / num_masked if num_masked > 0 else 0.0

            loss = mlm_loss
            
            # Backprop
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Logging
            total += 1
            ce_tot += mlm_loss.item()
            acc_tot += mlm_accuracy


        # Validation loop
        model.eval()
        total_val, ce_tot_val, acc_tot_val = 0, 0, 0

        for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            # Forward passes
            # Student (Bhili)
            for k, v in batch.items():
                batch[k] = v.to(DEVICE, non_blocking=True)
            
            inputs = {
                    "input_ids": batch["bh_input_ids"],
                    "attention_mask": batch["bh_attention_mask"]
                }
            with torch.no_grad():
                with autocast():
                    outputs = model(**inputs, output_hidden_states=True)
                    logits  = outputs.logits    # [B, T, V_student]
                    mlm_loss    = ce_loss_fn(
                        logits.view(-1, logits.size(-1)),
                        batch["bh_labels"].view(-1)
                    )
                
            # Compute MLM accuracy
            predictions = logits.argmax(dim=-1)  # [batch, seq_len]
            masked_positions = batch["bh_labels"] != -100  # [batch, seq_len]
            correct = (predictions == batch["bh_labels"]) & masked_positions  # [batch, seq_len]
            num_correct = correct.sum().item()
            num_masked = masked_positions.sum().item()
            mlm_accuracy = num_correct / num_masked if num_masked > 0 else 0.0


            # Logging
            total_val += 1
            ce_tot_val += mlm_loss.item()
            acc_tot_val += mlm_accuracy

        logging.info(
            f"[Epoch {epoch}]\n "
            f"Train Loss={ce_tot/total:.4f} Train Acc={acc_tot/total:.4f} | "
            f"Val Loss={ce_tot_val/total_val:.4f} Val Acc={acc_tot_val/total_val:.4f}"
        )
        
        # Save checkpoint
        if loss < best_val_loss:
            best_val_loss = loss
            # This saves model.safetensors (or pytorch_model.bin) and config.json
            model.save_pretrained(OUTPUT_DIR) 
            
            # This saves tokenizer_config.json, vocab.txt (or merges.txt for BPE, etc.)
            tokenizer.save_pretrained(OUTPUT_DIR) 
            
            logging.info(f"Saved best model and tokenizer to {OUTPUT_DIR} at epoch {epoch}")

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()