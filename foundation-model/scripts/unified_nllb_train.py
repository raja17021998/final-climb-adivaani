
import os
import math
import torch
import random
import numpy as np
import pandas as pd
import torch.distributed as dist

from collections import defaultdict
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)

from torch.utils.data import Sampler, DataLoader
from text_protection import protect_text
import config


# ======================================================
# 🔵 DDP INITIALIZATION (UNCHANGED)
# ======================================================
RANK = int(os.environ.get("RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

is_distributed = WORLD > 1

if is_distributed:
    dist.init_process_group(
        backend=config.DDP_BACKEND,
        timeout=config.DDP_TIMEOUT,
    )
    torch.cuda.set_device(LOCAL_RANK)
    dist.barrier()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = config.SEED + RANK
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ======================================================
# OUTPUT DIR (UNCHANGED)
# ======================================================
variant = config.MODEL_VARIANT.replace(".", "_")
suffix = "lora" if config.USE_LORA else "full"
OUT_DIR = os.path.join(config.OUTPUT_DIR, f"{variant}_{suffix}")
FINAL_WEIGHT_DIR = os.path.join(OUT_DIR, "final_model_weight")

if RANK == 0:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FINAL_WEIGHT_DIR, exist_ok=True)

if is_distributed:
    dist.barrier()


# ======================================================
# DATA LOADING (UNCHANGED)
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
# ROUTE INFERENCE (UNCHANGED)
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
        bucket = f"{tt}_{sh}_to_{tt}"
    elif st and th:
        bucket = f"{st}_{st}_to_{th}"
    else:
        return None

    if bucket not in config.VALID_BUCKETS:
        return None

    return bucket


def attach(ds):
    def _f(batch):
        bucket, keep = [], []
        for s, t in zip(batch[config.SOURCE_LANG_COL],
                        batch[config.TARGET_LANG_COL]):
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
# TOKENIZER / MODEL (UNCHANGED)
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(
    config.MODEL_MAP[config.MODEL_VARIANT]
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    config.MODEL_MAP[config.MODEL_VARIANT],
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
).to(device)


# ======================================================
# LANGUAGE TOKEN EXTENSION (FIXED)
# ======================================================
if config.ENABLE_LANG_TOKEN_EXTENSION and config.NEW_LANGUAGE_TOKEN_MAP:

    existing_vocab = tokenizer.get_vocab()

    new_tokens = [
        tok for tok in config.NEW_LANGUAGE_TOKEN_MAP
        if tok not in existing_vocab
    ]

    if new_tokens:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": new_tokens}
        )
        model.resize_token_embeddings(len(tokenizer))

    # --------------------------------------------------
    # 1️⃣ Copy initialization embeddings
    # --------------------------------------------------
    with torch.no_grad():
        for new_lang, init_lang in config.NEW_LANGUAGE_TOKEN_MAP.items():
            new_id = tokenizer.convert_tokens_to_ids(new_lang)
            init_id = tokenizer.convert_tokens_to_ids(init_lang)

            model.model.shared.weight[new_id] = (
                model.model.shared.weight[init_id].clone()
            )

    # --------------------------------------------------
    # 2️⃣ Register SINGLE gradient mask hook (once)
    # --------------------------------------------------
    if config.FREEZE_NEW_LANG_TOKENS:

        frozen_ids = [
            tokenizer.convert_tokens_to_ids(tok)
            for tok in config.NEW_LANGUAGE_TOKEN_MAP.keys()
        ]

        def grad_mask_hook(grad):
            # Zero out gradients for frozen token rows
            grad = grad.clone()
            grad[frozen_ids] = 0
            return grad

        model.model.shared.weight.register_hook(grad_mask_hook)


if is_distributed:
    dist.barrier()


# ======================================================
# PREPROCESS (UNCHANGED)
# ======================================================
def preprocess(batch):
    input_ids, masks, labels = [], [], []

    for s, t, sl, tl in zip(
        batch[config.SOURCE_COL],
        batch[config.TARGET_COL],
        batch[config.SOURCE_LANG_COL],
        batch[config.TARGET_LANG_COL],
    ):
        s, _ = protect_text(s)
        t, _ = protect_text(t)

        tokenizer.src_lang = sl
        tokenizer.tgt_lang = tl
        t = f"{tl} {t}"

        enc = tokenizer(s,
                        truncation=True,
                        padding="max_length",
                        max_length=config.MAX_LEN)

        dec = tokenizer(text_target=t,
                        truncation=True,
                        padding="max_length",
                        max_length=config.MAX_LEN)

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


def rebuild_weight_vector(device, dtype):
    global bucket_to_idx, weight_vector

    buckets_sorted = sorted(lang_states.keys())

    bucket_to_idx = {b: i for i, b in enumerate(buckets_sorted)}

    weight_vector = torch.tensor(
        [lang_states[b].weight for b in buckets_sorted],
        device=device,
        dtype=dtype,
    )

train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
val_ds = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)


# ======================================================
# LOSS STATE
# ======================================================
class LangState:
    def __init__(self):
        self.best = float("inf")
        self.patience = 0
        self.weight = 1.0


lang_states = defaultdict(LangState)
bucket_to_idx = {}
weight_vector = None


# Initialize lang_states for all training buckets
for b in set(train_ds["bucket"]):
    _ = lang_states[b]

# Build initial weight vector
rebuild_weight_vector(
    device=device,
    dtype=next(model.parameters()).dtype,
)


# ======================================================
# 🔧 NON-REPLACEMENT TEMPERATURE SAMPLER (DDP SAFE)
# ======================================================
class DistributedTemperatureSampler(Sampler):

    def __init__(self, dataset, temperature):
        self.dataset = dataset
        self.temperature = temperature
        self.rank = RANK
        self.world_size = WORLD
        self.epoch = 0

        bucket_counts = defaultdict(int)
        for b in dataset["bucket"]:
            bucket_counts[b] += 1

        alpha = 1.0 / temperature
        bucket_scores = {b: (c ** alpha) for b, c in bucket_counts.items()}
        total = sum(bucket_scores.values())
        self.bucket_probs = {b: s / total for b, s in bucket_scores.items()}

        self.num_samples = math.ceil(len(dataset) / self.world_size)
        self.total_size = self.num_samples * self.world_size

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(config.SEED + 1000 * self.epoch)

        # Shuffle indices
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # Reweight ordering by bucket priority (no replacement)
        indices.sort(
            key=lambda i: self.bucket_probs[self.dataset[i]["bucket"]],
            reverse=True
        )

        # Pad to total_size
        if len(indices) < self.total_size:
            padding = indices[: self.total_size - len(indices)]
            indices += padding

        # Shard
        indices = indices[self.rank:self.total_size:self.world_size]

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


train_sampler = DistributedTemperatureSampler(train_ds, config.TEMPERATURE)


# ======================================================
# COLLATOR (UNCHANGED)
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
# TRAINER (FIXED – teacher forcing safe)
# ======================================================
class WeightedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        buckets = inputs.pop("bucket")
        labels = inputs.pop("labels")

        # decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)

        outputs = model(
            **inputs,
            labels=labels,
        )


        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction="none",
            ignore_index=-100,
        )

        vocab_size = shift_logits.size(-1)

        loss = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
        )

        loss = loss.view(shift_labels.size())

        token_counts = (shift_labels != -100).sum(dim=1)
        sample_loss = loss.sum(dim=1) / (token_counts + 1e-8)

        bucket_indices = torch.tensor(
            [bucket_to_idx[b] for b in buckets],
            device=sample_loss.device,
        )

        weights = weight_vector[bucket_indices]

        weighted_loss = (sample_loss * weights).mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss


    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )


def get_direction(bucket):
    """
    Derive high-level direction from bucket name.
    Example:
        bhili_hindi_to_bhili  -> hub_to_tribal
        bhili_bhili_to_hindi  -> tribal_to_hub
    """
    try:
        left, right = bucket.split("_to_")

        src_lang = left.split("_")[-1]
        tgt_lang = right

        if src_lang in config.ALL_HUB and tgt_lang in config.ALL_TRIBAL:
            return "hub_to_tribal"

        if src_lang in config.ALL_TRIBAL and tgt_lang in config.ALL_HUB:
            return "tribal_to_hub"

    except Exception:
        pass

    return "unknown"

# ======================================================
# 🔧 PERFECTLY SHARDED EVAL + FULL STATE SYNC
# ======================================================
class PatienceCallback(TrainerCallback):

    def on_evaluate(self, args, state, control, **kwargs):

        model = kwargs["model"]
        eval_dataset = kwargs["eval_dataloader"].dataset
        model.eval()

        bucket_loss = defaultdict(list)

        # ==================================================
        # 1️⃣ Distributed Validation Sampler
        # ==================================================
        if is_distributed:
            from torch.utils.data import DistributedSampler

            val_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=WORLD,
                rank=RANK,
                shuffle=False,
                drop_last=False,
            )

            epoch = int(state.epoch) if state.epoch is not None else 0
            val_sampler.set_epoch(epoch)
        else:
            val_sampler = None

        # ==================================================
        # 2️⃣ Validation Loader
        # ==================================================
        val_loader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=BucketAwareCollator(tokenizer, model),
            num_workers=0,
            pin_memory=False,
        )

        # ==================================================
        # 3️⃣ Forward Pass (teacher forcing via labels)
        # ==================================================
        with torch.no_grad():
            for batch in val_loader:

                buckets = batch.pop("bucket")
                labels = batch.pop("labels")

                batch = {k: v.to(device) for k, v in batch.items()}
                labels = labels.to(device)

                outputs = model(**batch, labels=labels)

                # token-level loss (already shift-safe internally)
                logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(
                    reduction="none",
                    ignore_index=-100,
                )

                vocab_size = shift_logits.size(-1)

                loss = loss_fct(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                )

                loss = loss.view(shift_labels.size())

                token_counts = (shift_labels != -100).sum(dim=1)
                sample_loss = loss.sum(dim=1) / (token_counts + 1e-8)

                for b, l in zip(buckets, sample_loss.tolist()):
                    bucket_loss[b].append(l)

        # ==================================================
        # 4️⃣ Gather Across All Ranks
        # ==================================================
        if is_distributed:
            gathered = [None for _ in range(WORLD)]
            dist.all_gather_object(gathered, bucket_loss)

            merged = defaultdict(list)
            for part in gathered:
                for b in part:
                    merged[b].extend(part[b])

            bucket_loss = merged

        # ==================================================
        # 5️⃣ Rank 0 Updates Curriculum + Logs
        # ==================================================
        if RANK == 0:

            epoch_num = int(state.epoch) if state.epoch is not None else 0
            print(f"\n📊 [LangEval] Epoch {epoch_num}")

            for b in sorted(bucket_loss.keys()):

                if len(bucket_loss[b]) == 0:
                    continue

                avg = sum(bucket_loss[b]) / len(bucket_loss[b])
                state_obj = lang_states[b]

                # Curriculum logic
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

                direction = get_direction(b)

                print(
                    f"  {b:<35} | "
                    f"dir={direction:<15} | "
                    f"loss={avg:.4f} | "
                    f"weight={state_obj.weight:.3f}"
                )

        # ==================================================
        # 6️⃣ Broadcast Updated lang_states
        # ==================================================
        if is_distributed:

            if RANK == 0:
                state_dict = {
                    b: {
                        "best": lang_states[b].best,
                        "patience": lang_states[b].patience,
                        "weight": lang_states[b].weight,
                    }
                    for b in lang_states
                }
            else:
                state_dict = None

            container = [state_dict]
            dist.broadcast_object_list(container, src=0)
            state_dict = container[0]
            dist.barrier()

            lang_states.clear()

            for b, vals in state_dict.items():
                obj = LangState()
                obj.best = vals["best"]
                obj.patience = vals["patience"]
                obj.weight = vals["weight"]
                lang_states[b] = obj

            # Rebuild weight vector
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype

            rebuild_weight_vector(
                device=model_device,
                dtype=model_dtype,
            )



class SamplerCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else 0
        train_sampler.set_epoch(int(state.epoch))


# ======================================================
# 🔧 4️⃣ CHECKPOINT-SAFE STATE
# ======================================================
class LangStateCheckpointCallback(TrainerCallback):

    def on_save(self, args, state, control, **kwargs):
        if RANK != 0:
            return

        save_path = os.path.join(args.output_dir, "lang_states.pt")

        state_dict = {
            b: {
                "best": lang_states[b].best,
                "patience": lang_states[b].patience,
                "weight": lang_states[b].weight,
            }
            for b in lang_states
        }

        torch.save(state_dict, save_path)

    def on_train_begin(self, args, state, control, **kwargs):

            load_path = os.path.join(args.output_dir, "lang_states.pt")

            state_dict = None

            # Only rank 0 reads from disk
            if RANK == 0 and os.path.exists(load_path):
                state_dict = torch.load(load_path, map_location="cpu")
                print("🔁 Restored lang_states from checkpoint.")

            if is_distributed:
                container = [state_dict]
                dist.broadcast_object_list(container, src=0)
                state_dict = container[0]
                dist.barrier()

            # If state_dict exists (after broadcast), rebuild state
            if state_dict is not None:
                lang_states.clear()

                for b, vals in state_dict.items():
                    obj = LangState()
                    obj.best = vals["best"]
                    obj.patience = vals["patience"]
                    obj.weight = vals["weight"]
                    lang_states[b] = obj

                model_device = next(kwargs["model"].parameters()).device
                model_dtype = next(kwargs["model"].parameters()).dtype

                rebuild_weight_vector(
                    device=model_device,
                    dtype=model_dtype,
                )



# ======================================================
# TRAIN
# ======================================================
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
    callbacks=[
        SamplerCallback(),
        PatienceCallback(),
        LangStateCheckpointCallback(),
    ],
)

trainer.train()

if RANK == 0:
    trainer.save_model(FINAL_WEIGHT_DIR)
    tokenizer.save_pretrained(FINAL_WEIGHT_DIR)

if is_distributed:
    dist.barrier()
    dist.destroy_process_group()

print("✅ Fully hardened DDP training complete.")
