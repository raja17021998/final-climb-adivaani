# # train_ddp.py
# import os
# import json
# import torch
# import torch.distributed as dist
# from torch.utils.data import DataLoader, DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# import matplotlib.pyplot as plt
# from transformers import get_linear_schedule_with_warmup

# from config import *
# from dataset_utils import load_split_dataset, TranslationDataset
# from model_utils import build_model_and_tokenizer


# def setup_ddp():
#     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         dist.init_process_group("nccl")
#         local_rank = int(os.environ["LOCAL_RANK"])
#         torch.cuda.set_device(local_rank)
#         return True, local_rank, dist.get_rank()
#     return False, 0, 0


# def train():
#     is_ddp, local_rank, rank = setup_ddp()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     for tribal in TRIBAL_LANGS:
#         hp = HYPERPARAMS[tribal]

#         tokenizer, model = build_model_and_tokenizer(tribal, rank)
#         model.to(device)

#         if is_ddp:
#             model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#             model = DDP(model, device_ids=[local_rank], output_device=local_rank)

#         hubs = HUB_LANGS_BHILI if tribal == "bhili" else HUB_LANGS_COMMON

#         for hub in hubs:
#             for src, tgt in [(hub, tribal), (tribal, hub)]:
#                 direction_key = f"{src}_{tgt}"

#                 # Direction selection from config
#                 if DIRECTION_CONFIG.get(tribal, {}).get(direction_key, True) is False:
#                     continue

#                 train_df, val_df = load_split_dataset(tribal)

#                 # Apply per-direction data limit
#                 limit = DIRECTION_DATA_LIMIT.get(tribal, {}).get(direction_key, None)
#                 if limit is not None:
#                     train_df = train_df.head(limit)

#                 src_col = LANGUAGE_COLUMN_MAP[tribal][src]
#                 tgt_col = LANGUAGE_COLUMN_MAP[tribal][tgt]

#                 train_ds = TranslationDataset(train_df, src_col, tgt_col, tokenizer, hp["max_length"])
#                 val_ds = TranslationDataset(val_df, src_col, tgt_col, tokenizer, hp["max_length"])

#                 if is_ddp:
#                     train_sampler = DistributedSampler(train_ds)
#                     val_sampler = DistributedSampler(val_ds, shuffle=False)
#                 else:
#                     train_sampler = None
#                     val_sampler = None

#                 train_loader = DataLoader(
#                     train_ds,
#                     batch_size=hp["batch_size"],
#                     sampler=train_sampler,
#                     shuffle=train_sampler is None,
#                 )

#                 val_loader = DataLoader(
#                     val_ds,
#                     batch_size=hp["batch_size"],
#                     sampler=val_sampler,
#                 )

#                 optimizer = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"])
#                 scheduler = get_linear_schedule_with_warmup(
#                     optimizer,
#                     hp["warmup_steps"],
#                     hp["num_epochs"] * len(train_loader),
#                 )

#                 train_losses, val_losses = [], []

#                 for epoch in range(hp["num_epochs"]):
#                     if is_ddp and train_sampler is not None:
#                         train_sampler.set_epoch(epoch)

#                     # =======================
#                     # Training
#                     # =======================
#                     model.train()
#                     total_loss = 0.0

#                     for batch in train_loader:
#                         batch = {k: v.to(device) for k, v in batch.items()}

#                         outputs = model(**batch)
#                         loss = outputs.loss
#                         loss.backward()

#                         optimizer.step()
#                         scheduler.step()
#                         optimizer.zero_grad()

#                         total_loss += loss.item()

#                     train_loss = total_loss / len(train_loader)

#                     # =======================
#                     # Validation
#                     # =======================
#                     model.eval()
#                     val_loss = 0.0

#                     with torch.no_grad():
#                         for batch in val_loader:
#                             batch = {k: v.to(device) for k, v in batch.items()}
#                             outputs = model(**batch)
#                             val_loss += outputs.loss.item()

#                     val_loss /= len(val_loader)

#                     # =======================
#                     # Sync losses across GPUs
#                     # =======================
#                     if is_ddp:
#                         tl = torch.tensor(train_loss, device=device)
#                         vl = torch.tensor(val_loss, device=device)
#                         dist.all_reduce(tl)
#                         dist.all_reduce(vl)
#                         train_loss = tl.item() / dist.get_world_size()
#                         val_loss = vl.item() / dist.get_world_size()

#                     if rank == 0:
#                         print(
#                             f"[{tribal}] {src}->{tgt} | Epoch {epoch+1} | "
#                             f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
#                         )

#                     train_losses.append(train_loss)
#                     val_losses.append(val_loss)

#                 # =======================
#                 # Save model + logs (rank 0)
#                 # =======================
#                 if rank == 0:
#                     model_name = MODEL_CHOICE[tribal].split("/")[-1]
#                     tag = "lora" if USE_LORA[tribal] else "normal"
#                     direction = f"{src}_{tgt}"

#                     save_dir = os.path.join(MODEL_SAVE_DIR, f"{model_name}_{tag}", tribal, direction)
#                     os.makedirs(save_dir, exist_ok=True)

#                     (model.module if is_ddp else model).save_pretrained(save_dir)
#                     tokenizer.save_pretrained(save_dir)

#                     plot_dir = os.path.join(PLOTS_DIR, f"{model_name}_{tag}", tribal, direction)
#                     log_dir = os.path.join(LOGS_DIR, f"{model_name}_{tag}", tribal, direction)
#                     os.makedirs(plot_dir, exist_ok=True)
#                     os.makedirs(log_dir, exist_ok=True)

#                     # Plot losses
#                     plt.figure()
#                     plt.plot(train_losses, label="train")
#                     plt.plot(val_losses, label="val")
#                     plt.xlabel("Epoch")
#                     plt.ylabel("Loss")
#                     plt.legend()
#                     plt.tight_layout()
#                     plt.savefig(os.path.join(plot_dir, "loss.png"))
#                     plt.close()

#                     # Save logs
#                     with open(os.path.join(log_dir, "loss.json"), "w") as f:
#                         json.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

#         if is_ddp:
#             dist.barrier()


# if __name__ == "__main__":
#     train()


# train_ddp.py
import os
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from config import *
from dataset_utils import load_split_dataset, TranslationDataset
from model_utils import build_model_and_tokenizer


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank, dist.get_rank()
    return False, 0, 0


def train():
    is_ddp, local_rank, rank = setup_ddp()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for tribal in TRIBAL_LANGS:
        hp = HYPERPARAMS[tribal]

        tokenizer, model = build_model_and_tokenizer(tribal, rank)
        model.to(device)

        if is_ddp:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        hubs = HUB_LANGS_BHILI if tribal == "bhili" else HUB_LANGS_COMMON

        for hub in hubs:
            for src, tgt in [(hub, tribal), (tribal, hub)]:
                direction_key = f"{src}_{tgt}"

                if DIRECTION_CONFIG.get(tribal, {}).get(direction_key, True) is False:
                    continue

                train_df, val_df = load_split_dataset(tribal)

                limit = DIRECTION_DATA_LIMIT.get(tribal, {}).get(direction_key, None)
                if limit is not None:
                    train_df = train_df.head(limit)

                src_col = LANGUAGE_COLUMN_MAP[tribal][src]
                tgt_col = LANGUAGE_COLUMN_MAP[tribal][tgt]

                train_ds = TranslationDataset(train_df, src_col, tgt_col, tokenizer, hp["max_length"])
                val_ds = TranslationDataset(val_df, src_col, tgt_col, tokenizer, hp["max_length"])

                if is_ddp:
                    train_sampler = DistributedSampler(train_ds)
                    val_sampler = DistributedSampler(val_ds, shuffle=False)
                else:
                    train_sampler = None
                    val_sampler = None

                train_loader = DataLoader(
                    train_ds,
                    batch_size=hp["batch_size"],
                    sampler=train_sampler,
                    shuffle=train_sampler is None,
                )

                val_loader = DataLoader(
                    val_ds,
                    batch_size=hp["batch_size"],
                    sampler=val_sampler,
                )

                if rank == 0:
                    print(
                        f"\n==== Training Direction: {src} -> {tgt} | Tribal: {tribal} ===="
                    )
                    print(
                        f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}"
                    )

                optimizer = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"])
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    hp["warmup_steps"],
                    hp["num_epochs"] * len(train_loader),
                )

                train_losses, val_losses = [], []

                for epoch in range(hp["num_epochs"]):
                    if is_ddp and train_sampler is not None:
                        train_sampler.set_epoch(epoch)

                    # =======================
                    # Training
                    # =======================
                    model.train()
                    total_loss = 0.0

                    train_bar = tqdm(
                        train_loader,
                        desc=f"[Train][{tribal} {src}->{tgt}] Epoch {epoch+1}",
                        disable=rank != 0,
                    )

                    for batch in train_bar:
                        batch = {k: v.to(device) for k, v in batch.items()}

                        outputs = model(**batch)
                        loss = outputs.loss
                        loss.backward()

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        total_loss += loss.item()
                        if rank == 0:
                            train_bar.set_postfix(loss=loss.item())

                    train_loss = total_loss / len(train_loader)

                    # =======================
                    # Validation
                    # =======================
                    model.eval()
                    val_loss = 0.0

                    val_bar = tqdm(
                        val_loader,
                        desc=f"[Val][{tribal} {src}->{tgt}] Epoch {epoch+1}",
                        disable=rank != 0,
                    )

                    with torch.no_grad():
                        for batch in val_bar:
                            batch = {k: v.to(device) for k, v in batch.items()}
                            outputs = model(**batch)
                            loss = outputs.loss
                            val_loss += loss.item()
                            if rank == 0:
                                val_bar.set_postfix(loss=loss.item())

                    val_loss /= len(val_loader)

                    if is_ddp:
                        tl = torch.tensor(train_loss, device=device)
                        vl = torch.tensor(val_loss, device=device)
                        dist.all_reduce(tl)
                        dist.all_reduce(vl)
                        train_loss = tl.item() / dist.get_world_size()
                        val_loss = vl.item() / dist.get_world_size()

                    if rank == 0:
                        print(
                            f"[{tribal}] {src}->{tgt} | Epoch {epoch+1} | "
                            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                        )

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)

                if rank == 0:
                    model_name = MODEL_CHOICE[tribal].split("/")[-1]
                    tag = "lora" if USE_LORA[tribal] else "normal"
                    direction = f"{src}_{tgt}"

                    save_dir = os.path.join(MODEL_SAVE_DIR, f"{model_name}_{tag}", tribal, direction)
                    os.makedirs(save_dir, exist_ok=True)

                    (model.module if is_ddp else model).save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)

                    plot_dir = os.path.join(PLOTS_DIR, f"{model_name}_{tag}", tribal, direction)
                    log_dir = os.path.join(LOGS_DIR, f"{model_name}_{tag}", tribal, direction)
                    os.makedirs(plot_dir, exist_ok=True)
                    os.makedirs(log_dir, exist_ok=True)

                    plt.figure()
                    plt.plot(train_losses, label="train")
                    plt.plot(val_losses, label="val")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, "loss.png"))
                    plt.close()

                    with open(os.path.join(log_dir, "loss.json"), "w") as f:
                        json.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

        if is_ddp:
            dist.barrier()


if __name__ == "__main__":
    train()
