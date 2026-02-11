# import os
# import argparse
# import torch
# import pandas as pd
# import csv
# import json
# import matplotlib.pyplot as plt

# from collections import defaultdict
# from datetime import datetime
# from datasets import Dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSeq2SeqLM,
#     Trainer,
#     TrainingArguments,
#     DataCollatorForSeq2Seq,
#     TrainerCallback,
# )
# from sacrebleu import corpus_bleu, corpus_chrf

# from text_protection import protect_text

# # ======================================================
# # PATHS
# # ======================================================
# BASE_DIR = "/home/jovyan/final-climb-shashwat-do-not-delete/foundation-model"
# DATA_DIR = "/home/jovyan/final-climb-shashwat-do-not-delete/datasets/foundation-model"

# # ======================================================
# # MODEL MAP
# # ======================================================
# MODEL_MAP = {
#     "nllb-200-600m": "facebook/nllb-200-distilled-600M",
#     "nllb-200-1.3B": "facebook/nllb-200-1.3B",
#     "nllb-200-3.3B": "facebook/nllb-200-3.3B",
# }

# # ======================================================
# # ARGS
# # ======================================================
# parser = argparse.ArgumentParser()
# parser.add_argument("--mode", choices=["debug", "train"], default="train")
# parser.add_argument("--debug_rows", type=int, default=200)
# parser.add_argument("--epochs", type=int, default=10)
# parser.add_argument("--batch_size", type=int, default=4)
# parser.add_argument("--lr", type=float, default=1e-5)
# parser.add_argument("--seed", type=int, default=42)
# parser.add_argument("--model_variant", choices=list(MODEL_MAP.keys()), required=True)
# parser.add_argument("--use_lora", action="store_true")

# args = parser.parse_args()

# # ======================================================
# # VALIDATION
# # ======================================================
# if args.use_lora and args.model_variant != "nllb-200-3.3B":
#     raise ValueError("--use_lora only supported for nllb-200-3.3B")

# torch.manual_seed(args.seed)

# # ======================================================
# # DEVICE
# # ======================================================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"🚀 Using device: {device}")
# if device.type == "cuda":
#     print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")

# # ======================================================
# # OUTPUT DIR
# # ======================================================
# variant_name = args.model_variant.replace(".", "_")
# suffix = "lora" if args.use_lora else "full"

# OUT_DIR = os.path.join(BASE_DIR, "outputs", f"{variant_name}_{suffix}")
# os.makedirs(OUT_DIR, exist_ok=True)

# # ======================================================
# # DATA LOADING
# # ======================================================
# def load_csv(path, limit=None):
#     df = pd.read_csv(path)
#     if limit:
#         df = df.sample(n=limit, random_state=args.seed)
#     return Dataset.from_pandas(df.reset_index(drop=True))

# train_ds = load_csv(
#     os.path.join(DATA_DIR, "train.csv"),
#     args.debug_rows if args.mode == "debug" else None,
# )
# val_ds = load_csv(
#     os.path.join(DATA_DIR, "val.csv"),
#     args.debug_rows if args.mode == "debug" else None,
# )

# # ======================================================
# # METADATA DERIVATION
# # ======================================================
# TRIBAL_PREFIXES = {
#     "bhi": "bhili",
#     "mun": "mundari",
#     "gon": "gondi",
#     "san": "santali",
#     "gar": "garo",
#     "kui": "kuii",
# }


# def infer_tribal(lang):
#     for k, v in TRIBAL_PREFIXES.items():
#         if lang.startswith(k):
#             return v
#     return None


# def infer_hub_lang(lang):
#     if lang.startswith("hin"):
#         return "hindi"
#     if lang.startswith("eng"):
#         return "english"
#     return None


# def infer_lang_name(lang):
#     """
#     Returns canonical language name for logging/plots.
#     """
#     tribal = infer_tribal(lang)
#     if tribal:
#         return tribal

#     hub = infer_hub_lang(lang)
#     if hub:
#         return hub

#     return "other"


# def infer_direction(src, tgt):
#     src_name = infer_lang_name(src)
#     tgt_name = infer_lang_name(tgt)
#     return f"{src_name}_to_{tgt_name}"


# def attach_metadata(ds):
#     def _add(batch):
#         tribal = []
#         direction = []
#         for s, t in zip(batch["source_lang"], batch["target_lang"]):
#             tribal.append(infer_tribal(s) or infer_tribal(t))
#             direction.append(infer_direction(s, t))
#         batch["tribal_lang"] = tribal
#         batch["direction"] = direction
#         return batch
#     return ds.map(_add, batched=True)

# train_ds = attach_metadata(train_ds)
# val_ds = attach_metadata(val_ds)


# class MetadataSafeDataCollator(DataCollatorForSeq2Seq):
#     def __call__(self, features):
#         for f in features:
#             f.pop("source_sentence", None)
#             f.pop("target_sentence", None)
#             f.pop("source_lang", None)
#             f.pop("target_lang", None)
#             f.pop("tribal_lang", None)
#             f.pop("direction", None)
#             f.pop("dataset", None)   # ✅ REQUIRED
#         return super().__call__(features)



# # ======================================================
# # MODEL & TOKENIZER
# # ======================================================
# MODEL_ID = MODEL_MAP[args.model_variant]
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# model = AutoModelForSeq2SeqLM.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
# )
# model.to(device)

# # ======================================================
# # PREPROCESSING (UNCHANGED)
# # ======================================================
# MAX_LEN = 128

# def preprocess(batch):
#     input_ids, masks, labels = [], [], []

#     for s, t, sl, tl in zip(
#         batch["source_sentence"],
#         batch["target_sentence"],
#         batch["source_lang"],
#         batch["target_lang"],
#     ):
#         s, _ = protect_text(s)
#         t, _ = protect_text(t)

#         tokenizer.src_lang = sl
#         tokenizer.tgt_lang = tl
#         t = f"{tl} {t}"

#         enc = tokenizer(s, truncation=True, padding="max_length", max_length=MAX_LEN)
#         dec = tokenizer(text_target=t, truncation=True, padding="max_length", max_length=MAX_LEN)

#         input_ids.append(enc["input_ids"])
#         masks.append(enc["attention_mask"])
#         labels.append([x if x != tokenizer.pad_token_id else -100 for x in dec["input_ids"]])

#     batch["input_ids"] = input_ids
#     batch["attention_mask"] = masks
#     batch["labels"] = labels
#     return batch

# train_ds = train_ds.map(preprocess, batched=True)
# val_ds = val_ds.map(preprocess, batched=True)

# # ======================================================
# # TRAINING ARGS
# # ======================================================
# training_args = TrainingArguments(
#     output_dir=OUT_DIR,
#     num_train_epochs=args.epochs,
#     per_device_train_batch_size=args.batch_size,
#     per_device_eval_batch_size=args.batch_size,
#     learning_rate=args.lr,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     save_total_limit=1,
#     bf16=(device.type == "cuda"),
#     report_to="none",
#     remove_unused_columns=False,  # ✅ REQUIRED for tribal_lang & direction in callbacks
# )


# # ======================================================
# # LANG-AWARE LOSS CALLBACK
# # ======================================================
# class LangAwareLossCallback(TrainerCallback):
#     def __init__(self, out_dir, eval_dataset, tokenizer):
#         self.eval_dataset = eval_dataset
#         self.tokenizer = tokenizer
#         self.lang_eval = defaultdict(list)

#         self.log_dir = os.path.join(out_dir, "logs")
#         os.makedirs(self.log_dir, exist_ok=True)
#         self.log_path = os.path.join(self.log_dir, "langwise_eval.log")

#         with open(self.log_path, "w") as f:
#             f.write(f"Lang-wise Eval Log started at {datetime.now()}\n\n")


#     def on_evaluate(self, args, state, control, **kwargs):
#         model = kwargs["model"]
#         model.eval()

#         bucket = defaultdict(list)

#         for ex in self.eval_dataset:
#             self.tokenizer.src_lang = ex["source_lang"]
#             self.tokenizer.tgt_lang = ex["target_lang"]

#             enc = self.tokenizer(
#                 ex["source_sentence"],
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=128,
#             ).to(model.device)

#             with torch.no_grad():
#                 out = model(
#                     **enc,
#                     labels=self.tokenizer(
#                         f"{ex['target_lang']} {ex['target_sentence']}",
#                         return_tensors="pt",
#                         truncation=True,
#                         max_length=128,
#                     ).input_ids.to(model.device),
#                 )

#             key = f"{ex['tribal_lang']}_{ex['direction']}"
#             bucket[key].append(out.loss.item())

#         print(f"\n📊 [LangEval] Epoch {state.epoch:.0f}")
#         with open(self.log_path, "a") as f:
#             f.write(f"[Epoch {state.epoch:.0f}]\n")

#             for k in sorted(bucket):
#                 avg = sum(bucket[k]) / len(bucket[k])
#                 self.lang_eval[k].append((state.epoch, avg))

#                 print(f"  {k:<30} : {avg:.4f}")
#                 f.write(f"{k:<30} : {avg:.4f}\n")

#             f.write("\n")


# loss_tracker = LangAwareLossCallback(
#     OUT_DIR,
#     eval_dataset=val_ds,
#     tokenizer=tokenizer,
# )


# # ======================================================
# # TRAINER
# # ======================================================
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_ds,
#     eval_dataset=val_ds,
#     # tokenizer=tokenizer,
#     data_collator=MetadataSafeDataCollator(tokenizer, model),
#     callbacks=[loss_tracker],
# )

# # ======================================================
# # TRAIN
# # ======================================================
# trainer.train()
# trainer.save_model(OUT_DIR)
# tokenizer.save_pretrained(OUT_DIR)

# # ======================================================
# # SAVE LANG LOSS CSV + PLOTS
# # ======================================================
# with open(os.path.join(OUT_DIR, "lang_eval_losses.csv"), "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["epoch", "lang_direction", "loss"])
#     for k, vals in loss_tracker.lang_eval.items():
#         for e, l in vals:
#             writer.writerow([e, k, l])

# for k, vals in loss_tracker.lang_eval.items():
#     plt.figure()
#     plt.plot([x[0] for x in vals], [x[1] for x in vals], marker="o")
#     plt.title(f"Eval Loss: {k}")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.grid(True)
#     plt.savefig(os.path.join(OUT_DIR, f"loss_{k}.png"), dpi=150)
#     plt.close()

# # ======================================================
# # BLEU + chrF (UNCHANGED)
# # ======================================================
# def evaluate_bleu_chrf(model, tokenizer, dataset):
#     scores = defaultdict(lambda: {"refs": [], "hyps": []})

#     model.eval()
#     for ex in dataset:
#         tokenizer.src_lang = ex["source_lang"]
#         tokenizer.tgt_lang = ex["target_lang"]

#         inp = tokenizer(ex["source_sentence"], return_tensors="pt").to(model.device)
#         gen = model.generate(**inp, max_length=128)
#         pred = tokenizer.decode(gen[0], skip_special_tokens=True)

#         key = f"{ex['tribal_lang']}_{ex['direction']}"
#         scores[key]["refs"].append(ex["target_sentence"])
#         scores[key]["hyps"].append(pred)

#     out = {}
#     for k, v in scores.items():
#         out[k] = {
#             "BLEU": corpus_bleu(v["hyps"], [v["refs"]]).score,
#             "chrF": corpus_chrf(v["hyps"], [v["refs"]]).score,
#         }
#     return out

# metrics = evaluate_bleu_chrf(model, tokenizer, val_ds)

# with open(os.path.join(OUT_DIR, "bleu_chrf.json"), "w") as f:
#     json.dump(metrics, f, indent=2)

# print("✅ Training + hub-aware language-wise evaluation complete.")
# print(f"📁 Results saved to {OUT_DIR}")


import os
import argparse
import torch
import pandas as pd
import csv
import json
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
from sacrebleu import corpus_bleu, corpus_chrf

from text_protection import protect_text

# ======================================================
# PATHS
# ======================================================
BASE_DIR = "/home/jovyan/final-climb-shashwat-do-not-delete/foundation-model"
DATA_DIR = "/home/jovyan/final-climb-shashwat-do-not-delete/datasets/foundation-model"

# ======================================================
# MODEL MAP
# ======================================================
MODEL_MAP = {
    "nllb-200-600m": "facebook/nllb-200-distilled-600M",
    "nllb-200-1.3B": "facebook/nllb-200-1.3B",
    "nllb-200-3.3B": "facebook/nllb-200-3.3B",
}

# ======================================================
# ARGS
# ======================================================
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["debug", "train"], default="train")
parser.add_argument("--debug_rows", type=int, default=200)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_variant", choices=list(MODEL_MAP.keys()), required=True)
parser.add_argument("--use_lora", action="store_true")

args = parser.parse_args()

# ======================================================
# VALIDATION
# ======================================================
if args.use_lora and args.model_variant != "nllb-200-3.3B":
    raise ValueError("--use_lora only supported for nllb-200-3.3B")

torch.manual_seed(args.seed)

# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if device.type == "cuda":
    print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")

# ======================================================
# OUTPUT DIR
# ======================================================
variant_name = args.model_variant.replace(".", "_")
suffix = "lora" if args.use_lora else "full"

OUT_DIR = os.path.join(BASE_DIR, "outputs", f"{variant_name}_{suffix}")
os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# DATA LOADING
# ======================================================
def load_csv(path, limit=None):
    df = pd.read_csv(path)
    if limit:
        df = df.sample(n=limit, random_state=args.seed)
    return Dataset.from_pandas(df.reset_index(drop=True))

train_ds = load_csv(
    os.path.join(DATA_DIR, "train.csv"),
    args.debug_rows if args.mode == "debug" else None,
)
val_ds = load_csv(
    os.path.join(DATA_DIR, "val.csv"),
    args.debug_rows if args.mode == "debug" else None,
)

# ======================================================
# METADATA DERIVATION (UPDATED – CANONICAL & DATA-DRIVEN)
# ======================================================
TRIBAL_PREFIXES = {
    "bhi": "bhili",
    "mun": "mundari",
    "gon": "gondi",
    "san": "santali",
    "gar": "garo",
    "kui": "kuii",
}

HUB_PREFIX_MAP = {
    "hin": "hindi",
    "eng": "english",
    "mar": "marathi",
    "guj": "gujarati",
}

def infer_tribal(lang):
    for k, v in TRIBAL_PREFIXES.items():
        if lang.startswith(k):
            return v
    return None

def infer_hub(lang):
    for k, v in HUB_PREFIX_MAP.items():
        if lang.startswith(k):
            return v
    return None

def infer_allowed_hubs_from_dataset(ds):
    hubs = set()
    for ex in ds:
        for lang in (ex["source_lang"], ex["target_lang"]):
            hub = infer_hub(lang)
            if hub:
                hubs.add(hub)
    return hubs

ALLOWED_HUBS = infer_allowed_hubs_from_dataset(train_ds)
print(f"✅ Allowed hubs inferred from data: {sorted(ALLOWED_HUBS)}")

def infer_canonical_direction(src, tgt):
    src_tribal = infer_tribal(src)
    tgt_tribal = infer_tribal(tgt)

    src_hub = infer_hub(src)
    tgt_hub = infer_hub(tgt)

    # tribal -> hub
    if src_tribal and tgt_hub in ALLOWED_HUBS:
        return f"{src_tribal}_to_{tgt_hub}"

    # hub -> tribal
    if src_hub in ALLOWED_HUBS and tgt_tribal:
        return f"{src_hub}_to_{tgt_tribal}"

    return None

def attach_metadata(ds):
    def _add(batch):
        tribal = []
        direction = []
        keep = []

        for s, t in zip(batch["source_lang"], batch["target_lang"]):
            d = infer_canonical_direction(s, t)

            if d is None:
                keep.append(False)
                tribal.append(None)
                direction.append(None)
            else:
                keep.append(True)
                tribal.append(infer_tribal(s) or infer_tribal(t))
                direction.append(d)

        batch["tribal_lang"] = tribal
        batch["direction"] = direction
        batch["_keep"] = keep
        return batch

    ds = ds.map(_add, batched=True)
    ds = ds.filter(lambda x: x["_keep"])
    return ds

train_ds = attach_metadata(train_ds)
val_ds = attach_metadata(val_ds)

# ======================================================
# DATA COLLATOR (UNCHANGED)
# ======================================================
class MetadataSafeDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        for f in features:
            f.pop("source_sentence", None)
            f.pop("target_sentence", None)
            f.pop("source_lang", None)
            f.pop("target_lang", None)
            f.pop("tribal_lang", None)
            f.pop("direction", None)
            f.pop("dataset", None)
        return super().__call__(features)

# ======================================================
# MODEL & TOKENIZER
# ======================================================
MODEL_ID = MODEL_MAP[args.model_variant]
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
)
model.to(device)

# ======================================================
# PREPROCESSING (UNCHANGED)
# ======================================================
MAX_LEN = 128

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

        enc = tokenizer(s, truncation=True, padding="max_length", max_length=MAX_LEN)
        dec = tokenizer(text_target=t, truncation=True, padding="max_length", max_length=MAX_LEN)

        input_ids.append(enc["input_ids"])
        masks.append(enc["attention_mask"])
        labels.append([x if x != tokenizer.pad_token_id else -100 for x in dec["input_ids"]])

    batch["input_ids"] = input_ids
    batch["attention_mask"] = masks
    batch["labels"] = labels
    return batch

train_ds = train_ds.map(preprocess, batched=True)
val_ds = val_ds.map(preprocess, batched=True)

# ======================================================
# TRAINING ARGS (UNCHANGED)
# ======================================================
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=args.lr,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=1,
    bf16=(device.type == "cuda"),
    report_to="none",
    remove_unused_columns=False,
)

# ======================================================
# LANG-AWARE LOSS CALLBACK (UNCHANGED)
# ======================================================
class LangAwareLossCallback(TrainerCallback):
    def __init__(self, out_dir, eval_dataset, tokenizer):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.lang_eval = defaultdict(list)

        self.log_dir = os.path.join(out_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "langwise_eval.log")

        with open(self.log_path, "w") as f:
            f.write(f"Lang-wise Eval Log started at {datetime.now()}\n\n")

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()

        bucket = defaultdict(list)

        for ex in self.eval_dataset:
            self.tokenizer.src_lang = ex["source_lang"]
            self.tokenizer.tgt_lang = ex["target_lang"]

            enc = self.tokenizer(
                ex["source_sentence"],
                return_tensors="pt",
                truncation=True,
                max_length=128,
            ).to(model.device)

            with torch.no_grad():
                out = model(
                    **enc,
                    labels=self.tokenizer(
                        f"{ex['target_lang']} {ex['target_sentence']}",
                        return_tensors="pt",
                        truncation=True,
                        max_length=128,
                    ).input_ids.to(model.device),
                )

            key = f"{ex['tribal_lang']}_{ex['direction']}"
            bucket[key].append(out.loss.item())

        print(f"\n📊 [LangEval] Epoch {state.epoch:.0f}")
        with open(self.log_path, "a") as f:
            f.write(f"[Epoch {state.epoch:.0f}]\n")

            for k in sorted(bucket):
                avg = sum(bucket[k]) / len(bucket[k])
                self.lang_eval[k].append((state.epoch, avg))

                print(f"  {k:<30} : {avg:.4f}")
                f.write(f"{k:<30} : {avg:.4f}\n")

            f.write("\n")

loss_tracker = LangAwareLossCallback(
    OUT_DIR,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
)

# ======================================================
# TRAINER
# ======================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=MetadataSafeDataCollator(tokenizer, model),
    callbacks=[loss_tracker],
)

# ======================================================
# TRAIN
# ======================================================
trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

# ======================================================
# SAVE LANG LOSS CSV + PLOTS (UNCHANGED)
# ======================================================
with open(os.path.join(OUT_DIR, "lang_eval_losses.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "lang_direction", "loss"])
    for k, vals in loss_tracker.lang_eval.items():
        for e, l in vals:
            writer.writerow([e, k, l])

for k, vals in loss_tracker.lang_eval.items():
    plt.figure()
    plt.plot([x[0] for x in vals], [x[1] for x in vals], marker="o")
    plt.title(f"Eval Loss: {k}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, f"loss_{k}.png"), dpi=150)
    plt.close()

# ======================================================
# BLEU + chrF (UNCHANGED)
# ======================================================
def evaluate_bleu_chrf(model, tokenizer, dataset):
    scores = defaultdict(lambda: {"refs": [], "hyps": []})

    model.eval()
    for ex in dataset:
        tokenizer.src_lang = ex["source_lang"]
        tokenizer.tgt_lang = ex["target_lang"]

        inp = tokenizer(ex["source_sentence"], return_tensors="pt").to(model.device)
        gen = model.generate(**inp, max_length=128)
        pred = tokenizer.decode(gen[0], skip_special_tokens=True)

        key = f"{ex['tribal_lang']}_{ex['direction']}"
        scores[key]["refs"].append(ex["target_sentence"])
        scores[key]["hyps"].append(pred)

    out = {}
    for k, v in scores.items():
        out[k] = {
            "BLEU": corpus_bleu(v["hyps"], [v["refs"]]).score,
            "chrF": corpus_chrf(v["hyps"], [v["refs"]]).score,
        }
    return out

metrics = evaluate_bleu_chrf(model, tokenizer, val_ds)

with open(os.path.join(OUT_DIR, "bleu_chrf.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Training + hub-aware language-wise evaluation complete.")
print(f"📁 Results saved to {OUT_DIR}")
