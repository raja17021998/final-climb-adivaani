import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import pandas as pd
from safetensors.torch import save_file
import sentencepiece as spm
from pathlib import Path

from bert_config import BertConfig
from model_bert import BertModel
from mlm_head import MLMHead
from dataset_mlm import MLMDataset

# ============================================================
# PATHS
# ============================================================
BASE_DIR = "/home/shashwat1/final-climb-shashwat-do-not-delete"
SAVE_ROOT = f"{BASE_DIR}/BERT"
TOKENIZER_PATH = f"{BASE_DIR}/tokenization/joint_spm.model"
CORPUS_PATH = f"{BASE_DIR}/tokenization/corpus.txt"

# ============================================================
# DDP SETUP
# ============================================================
def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return dist.get_rank(), dist.get_world_size()
    return 0, 1

# ============================================================
# TOKENIZER
# ============================================================
def load_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    return sp

# ============================================================
# SAFE PLOTTING (NO MATPLOTLIB CRASH)
# ============================================================
def safe_plot(logs, out_path, title):
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(
            [l["epoch"] for l in logs],
            [l["train_loss"] for l in logs]
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.grid()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"[WARN] Skipping plot due to matplotlib error: {e}")

# ============================================================
# TRAIN
# ============================================================
def train(lang, token_ids, epochs):
    rank, world = setup_ddp()
    device = torch.device("cuda")

    # -------- tokenizer semantics --------
    sp = load_tokenizer()
    pad_id  = sp.pad_id()              # <pad>
    cls_id  = sp.bos_id()              # <s>
    sep_id  = sp.eos_id()              # </s>
    mask_id = sp.piece_to_id("<mask>") # <mask>

    assert mask_id >= 0, "<mask> token not found in tokenizer"

    cfg = BertConfig(vocab_size=sp.get_piece_size())

    out_dir = f"{SAVE_ROOT}/{lang}"
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    dataset = MLMDataset(
        token_ids,
        cfg,
        cls=cls_id,
        sep=sep_id,
        mask=mask_id,
        pad=pad_id
    )

    sampler = DistributedSampler(dataset) if world > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=True
    )

    model = nn.ModuleDict({
        "bert": BertModel(cfg),
        "mlm": MLMHead(cfg),
    }).to(device)

    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()]
        )

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    autocast_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    logs = []

    # ========================================================
    # TRAIN LOOP
    # ========================================================
    for epoch in range(1, epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0

        pbar = tqdm(
            loader,
            desc=f"[{lang}] Epoch {epoch}/{epochs}",
            disable=(rank != 0)
        )

        for step, (ids, labels) in enumerate(pbar, 1):
            ids = ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                hidden = model["bert"](ids)
                logits = model["mlm"](hidden)
                loss = loss_fn(
                    logits.view(-1, cfg.vocab_size),
                    labels.view(-1)
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if rank == 0:
                pbar.set_postfix(train_loss=total_loss / step)

        train_loss = total_loss / len(loader)

        if rank == 0:
            logs.append({
                "epoch": epoch,
                "train_loss": train_loss
            })
            print(f"[{lang}] Epoch {epoch} | Train: {train_loss:.4f}")

    # ========================================================
    # SAVE ARTIFACTS (RANK 0 ONLY)
    # ========================================================
    if rank == 0:
        pd.DataFrame(logs).to_csv(
            f"{out_dir}/train_log_{lang.lower()}_bert.csv",
            index=False
        )

        safe_plot(
            logs,
            f"{out_dir}/loss_plot_{lang.lower()}_bert.png",
            f"{lang} BERT MLM"
        )

        cfg.save(f"{out_dir}/config.json")

        save_file(
            model.module.state_dict() if world > 1 else model.state_dict(),
            f"{out_dir}/model.safetensors"
        )

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="Limit number of corpus lines (debugging, like df[:7500])"
    )
    args = parser.parse_args()

    sp = load_tokenizer()

    sentences = Path(CORPUS_PATH).read_text().splitlines()
    if args.max_lines is not None:
        sentences = sentences[:args.max_lines]

    token_ids = [
        sp.encode(s)
        for s in sentences
        if len(s.strip()) > 0
    ]

    train(args.lang, token_ids, args.epochs)
