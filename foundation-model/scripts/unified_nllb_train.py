


import os
import torch
import pandas as pd
import torch.distributed as dist
import csv
import matplotlib.pyplot as plt

from collections import defaultdict
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)

from torch.utils.data import WeightedRandomSampler, DataLoader

from text_protection import protect_text
import config

# ======================================================
# DDP
# ======================================================
RANK = int(os.environ.get("RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))

if WORLD > 1 and not dist.is_initialized():
    dist.init_process_group(
        backend=config.DDP_BACKEND,
        timeout=config.DDP_TIMEOUT,
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(config.SEED)

# ======================================================
# OUTPUT DIR
# ======================================================
variant = config.MODEL_VARIANT.replace(".", "_")
suffix = "lora" if config.USE_LORA else "full"
OUT_DIR = os.path.join(config.OUTPUT_DIR, f"{variant}_{suffix}")

FINAL_WEIGHT_DIR = os.path.join(OUT_DIR, "final_model_weight")

if RANK == 0:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, config.LOG_DIR), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, config.LOSS_DIR), exist_ok=True)
    os.makedirs(FINAL_WEIGHT_DIR, exist_ok=True)

# ======================================================
# DATA LOADING
# ======================================================
def load_csv(path):
    df = pd.read_csv(path)
    if config.MODE == "debug":
        df = df.sample(
            n=min(len(df), config.DEBUG_ROWS),
            random_state=config.SEED,
        )
    return Dataset.from_pandas(df.reset_index(drop=True))

train_ds = load_csv(os.path.join(config.DATA_DIR, "train.csv"))
val_ds = load_csv(os.path.join(config.DATA_DIR, "val.csv"))

# ======================================================
# ROUTE INFERENCE
# ======================================================
def infer_lang(token, table):
    for name, prefixes in table.items():
        if any(token.startswith(p) for p in prefixes):
            return name
    return None

def infer_bucket(src, tgt):
    sh = infer_lang(src, config.HUB_LANGS)
    th = infer_lang(tgt, config.HUB_LANGS)
    st = infer_lang(src, config.TRIBAL_LANGS)
    tt = infer_lang(tgt, config.TRIBAL_LANGS)

    if sh and tt:
        return f"{tt}_{sh}_to_{tt}"
    if st and th:
        return f"{st}_{st}_to_{th}"
    return None

def attach(ds):
    def _f(batch):
        bucket = []
        keep = []

        for s, t in zip(batch["source_lang"], batch["target_lang"]):
            b = infer_bucket(s, t)
            bucket.append(b)
            keep.append(b is not None)

        batch["bucket"] = bucket
        batch["_keep"] = keep
        return batch

    ds = ds.map(_f, batched=True)
    ds = ds.filter(lambda x: x["_keep"])
    return ds

train_ds = attach(train_ds)
val_ds = attach(val_ds)

# ======================================================
# TOKENIZER / MODEL
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(
    config.MODEL_MAP[config.MODEL_VARIANT]
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    config.MODEL_MAP[config.MODEL_VARIANT],
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
).to(device)

# ======================================================
# PREPROCESS
# ======================================================
def preprocess(batch):
    input_ids, masks, labels = [], [], []

    for s, t, sl, tl in zip(
        batch["source_sentence"],
        batch["target_sentence"],
        batch["source_lang"],
        batch["target_lang"],
    ):
        s, _ = protect_text(s)
        t, _ = protect_text(t)

        tokenizer.src_lang = sl
        tokenizer.tgt_lang = tl
        t = f"{tl} {t}"

        enc = tokenizer(
            s,
            truncation=True,
            padding="max_length",
            max_length=config.MAX_LEN,
        )

        dec = tokenizer(
            text_target=t,
            truncation=True,
            padding="max_length",
            max_length=config.MAX_LEN,
        )

        input_ids.append(enc["input_ids"])
        masks.append(enc["attention_mask"])
        labels.append([
            x if x != tokenizer.pad_token_id else -100
            for x in dec["input_ids"]
        ])

    return {
        "input_ids": input_ids,
        "attention_mask": masks,
        "labels": labels,
        "bucket": batch["bucket"],
    }

train_ds = train_ds.map(
    preprocess,
    batched=True,
    remove_columns=train_ds.column_names,
)

val_ds = val_ds.map(
    preprocess,
    batched=True,
    remove_columns=val_ds.column_names,
)

# ======================================================
# 🆕 TEMPERATURE-BASED SAMPLING (ADDED SECTION)
# ======================================================
def build_temperature_sampler(dataset, temperature):

    bucket_counts = defaultdict(int)
    for b in dataset["bucket"]:
        bucket_counts[b] += 1

    alpha = 1.0 / temperature

    bucket_probs = {
        b: (count ** alpha)
        for b, count in bucket_counts.items()
    }

    total = sum(bucket_probs.values())

    bucket_probs = {
        b: p / total
        for b, p in bucket_probs.items()
    }

    sample_weights = [
        bucket_probs[b] for b in dataset["bucket"]
    ]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler

train_sampler = build_temperature_sampler(
    train_ds,
    config.TEMPERATURE,
)

# ======================================================
# LOSS STATE
# ======================================================
class LangState:
    def __init__(self):
        self.best = float("inf")
        self.patience = 0
        self.weight = 1.0

lang_states = defaultdict(LangState)

# ======================================================
# COLLATOR
# ======================================================
class BucketAwareCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        buckets = [f["bucket"] for f in features]
        for f in features:
            f.pop("bucket")
        batch = super().__call__(features)
        batch["bucket"] = buckets
        return batch

# ======================================================
# TRAINER
# ======================================================
class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        buckets = inputs.pop("bucket")
        outputs = model(**inputs)
        loss = outputs.loss

        weights = torch.tensor(
            [lang_states[b].weight for b in buckets],
            device=loss.device,
            dtype=loss.dtype,
        )

        weighted_loss = loss * weights.mean()
        return (weighted_loss, outputs) if return_outputs else weighted_loss

    # 🆕 Only addition: override dataloader to use sampler
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=4,
            pin_memory=True,
        )

# ======================================================
# CALLBACK
# ======================================================
class PatienceCallback(TrainerCallback):
    def __init__(self):
        self.lang_eval = defaultdict(list)

        self.log_dir = os.path.join(OUT_DIR, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "langwise_eval.log")

        if RANK == 0:
            with open(self.log_path, "w") as f:
                f.write(f"Lang-wise Eval Log started at {datetime.now()}\n\n")

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()

        bucket_loss = defaultdict(list)

        for ex in val_ds:
            enc = {
                "input_ids": torch.tensor([ex["input_ids"]]).to(device),
                "attention_mask": torch.tensor([ex["attention_mask"]]).to(device),
                "labels": torch.tensor([ex["labels"]]).to(device),
            }

            with torch.no_grad():
                out = model(**enc)

            bucket_loss[ex["bucket"]].append(out.loss.item())

        if RANK != 0:
            return

        print(f"\n📊 [LangEval] Epoch {int(state.epoch)}")

        with open(self.log_path, "a") as f:
            f.write(f"[Epoch {int(state.epoch)}]\n")

            for b in sorted(bucket_loss):
                avg = sum(bucket_loss[b]) / len(bucket_loss[b])
                state_obj = lang_states[b]

                if avg < state_obj.best:
                    state_obj.best = avg
                    state_obj.patience = 0
                else:
                    state_obj.patience += 1
                    if state_obj.patience >= config.PATIENCE:
                        state_obj.weight = max(
                            config.LOSS_FLOOR,
                            state_obj.weight * config.LOSS_DECAY,
                        )
                        state_obj.patience = 0

                self.lang_eval[b].append(
                    (state.epoch, avg, state_obj.weight)
                )

                print(
                    f"  {b:<35} | "
                    f"loss={avg:.4f} | "
                    f"weight={state_obj.weight:.3f}"
                )

                f.write(
                    f"{b:<35} | "
                    f"loss={avg:.4f} | "
                    f"weight={state_obj.weight:.3f}\n"
                )

            f.write("\n")

# ======================================================
# TRAIN
# ======================================================
patience_cb = PatienceCallback()

trainer = WeightedTrainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=(device.type == "cuda"),
        report_to="none",
        remove_unused_columns=False,
    ),
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=BucketAwareCollator(tokenizer, model),
    callbacks=[patience_cb],
)

trainer.train()

if RANK == 0:

    print("💾 Saving BEST model to final_model_weight/")
    trainer.save_model(FINAL_WEIGHT_DIR)
    tokenizer.save_pretrained(FINAL_WEIGHT_DIR)

    csv_path = os.path.join(OUT_DIR, "lang_eval_losses.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "bucket", "loss", "weight"])

        for b, vals in patience_cb.lang_eval.items():
            for e, l, w in vals:
                writer.writerow([e, b, l, w])

    for b, vals in patience_cb.lang_eval.items():
        plt.figure()
        plt.plot([x[0] for x in vals], [x[1] for x in vals], marker="o")
        plt.title(f"Eval Loss: {b}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(OUT_DIR, f"loss/loss_{b}.png"), dpi=150)
        plt.close()

print("✅ Training complete with per-language patience + weight decay + temperature sampling.")
