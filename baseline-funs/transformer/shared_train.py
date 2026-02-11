import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================= CONFIG =================

class Config:
    ROOT_DIR = "/home/user/Desktop/Shashwat/final-climb/baseline-funs"
    DATASET_DIR = "/home/user/Desktop/Shashwat/final-climb/datasets"
    SPM_PATH = os.path.join(ROOT_DIR, "joint_spm.model")
    SAVE_DIR = os.path.join(ROOT_DIR, "transformer", "shared")

    LANGS = ["bhili", "gondi", "kui", "mundari"]
    CSV_FILES = {
        "bhili": "Hin_Bhi_Mar_Guj.csv",
        "gondi": "Hin_Gon.csv",
        "kui": "Hin_Kui.csv",
        "mundari": "Hin_Mun.csv"
    }

    TRAIN_LIMITS = {
        "bhili": 5000,
        "gondi": 4000,
        "kui": 3000,
        "mundari": 5000
    }

    VAL_SPLIT = 0.1
    TEMP = 0.7   # 🔥 temperature for balanced sampling

    VOCAB_SIZE = 8000
    D_MODEL = 512
    N_HEADS = 8
    ENC_LAYERS = 6
    DEC_LAYERS = 6
    FF_DIM = 2048
    DROPOUT = 0.1

    BATCH_SIZE = 32
    LR = 3e-4
    EPOCHS = 5
    PAD_ID = 0

os.makedirs(Config.SAVE_DIR, exist_ok=True)

# ================= DISTRIBUTED SETUP =================

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        return rank, world_size, device, True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device, False

# ================= DATA =================

class LangDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.sp = spm.SentencePieceProcessor(model_file=Config.SPM_PATH)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        src = self.sp.encode(str(self.df.iloc[i, 0]), out_type=int, add_bos=True, add_eos=True)
        tgt = self.sp.encode(str(self.df.iloc[i, 1]), out_type=int, add_bos=True, add_eos=True)
        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    src, tgt = zip(*batch)
    return (
        nn.utils.rnn.pad_sequence(src, padding_value=Config.PAD_ID),
        nn.utils.rnn.pad_sequence(tgt, padding_value=Config.PAD_ID),
    )

# ================= POSITIONAL ENCODING =================

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, D)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ================= MODEL =================

class SharedTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(Config.VOCAB_SIZE, Config.D_MODEL)
        self.pe = SinusoidalPE(Config.D_MODEL)

        enc_layer = nn.TransformerEncoderLayer(
            Config.D_MODEL, Config.N_HEADS, Config.FF_DIM,
            dropout=Config.DROPOUT, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, Config.ENC_LAYERS)

        self.decoders = nn.ModuleDict()
        self.heads = nn.ModuleDict()

        for l in Config.LANGS:
            dec_layer = nn.TransformerDecoderLayer(
                Config.D_MODEL, Config.N_HEADS, Config.FF_DIM,
                dropout=Config.DROPOUT, batch_first=True
            )
            self.decoders[l] = nn.TransformerDecoder(dec_layer, Config.DEC_LAYERS)
            self.heads[l] = nn.Linear(Config.D_MODEL, Config.VOCAB_SIZE)

    def forward(self, src, tgt, lang):
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        memory = self.encoder(self.pe(self.emb(src)))
        out = self.decoders[lang](self.pe(self.emb(tgt[:, :-1])), memory)
        return self.heads[lang](out)

# ================= TEMP-BASED SAMPLING =================

def compute_sampling_probs(dataset_sizes, temperature):
    sizes = torch.tensor(dataset_sizes, dtype=torch.float)
    scaled = sizes.pow(1.0 / temperature)
    probs = scaled / scaled.sum()
    return probs

def build_language_schedule(langs, probs, steps, seed):
    g = torch.Generator().manual_seed(seed)
    indices = torch.multinomial(probs, steps, replacement=True, generator=g)
    return [langs[i] for i in indices.tolist()]

# ================= TRAIN =================

def main():
    rank, world_size, device, is_distributed = setup_distributed()

    train_loaders, val_loaders, samplers, sizes = {}, {}, {}, []

    for l in Config.LANGS:
        df = pd.read_csv(os.path.join(Config.DATASET_DIR, Config.CSV_FILES[l]))
        if Config.TRAIN_LIMITS.get(l):
            df = df.head(Config.TRAIN_LIMITS[l])

        split = int(len(df) * (1 - Config.VAL_SPLIT))
        train_df, val_df = df.iloc[:split], df.iloc[split:]

        train_ds = LangDataset(train_df)
        val_ds = LangDataset(val_df)

        sampler = DistributedSampler(train_ds) if is_distributed else None

        train_loaders[l] = DataLoader(
            train_ds, Config.BATCH_SIZE,
            sampler=sampler, shuffle=(sampler is None),
            collate_fn=collate_fn
        )
        val_loaders[l] = DataLoader(
            val_ds, Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
        )

        samplers[l] = sampler
        sizes.append(len(train_ds))

        if rank == 0:
            print(f"[DATA] {l} | Train={len(train_ds)} | Val={len(val_ds)}")

    probs = compute_sampling_probs(sizes, Config.TEMP)

    model = SharedTransformer().to(device)
    if is_distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_ID)

    history = []

    for epoch in range(Config.EPOCHS):
        model.train()
        for l in Config.LANGS:
            if samplers[l]:
                samplers[l].set_epoch(epoch)

        iters = {l: iter(train_loaders[l]) for l in Config.LANGS}
        total_steps = max(len(train_loaders[l]) for l in Config.LANGS) * len(Config.LANGS)

        schedule = build_language_schedule(
            Config.LANGS, probs, total_steps, seed=epoch + rank
        )

        train_loss = 0.0

        for lang in tqdm(schedule, disable=(rank != 0)):
            try:
                src, tgt = next(iters[lang])
            except StopIteration:
                iters[lang] = iter(train_loaders[lang])
                src, tgt = next(iters[lang])

            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            logits = model(src, tgt, lang)
            loss = criterion(
                logits.reshape(-1, Config.VOCAB_SIZE),
                tgt[1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= total_steps

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for l in Config.LANGS:
                for src, tgt in val_loaders[l]:
                    src, tgt = src.to(device), tgt.to(device)
                    logits = model(src, tgt, l)
                    loss = criterion(
                        logits.reshape(-1, Config.VOCAB_SIZE),
                        tgt[1:].reshape(-1)
                    )
                    val_loss += loss.item()

        val_loss /= sum(len(val_loaders[l]) for l in Config.LANGS)

        if rank == 0:
            print(f"Epoch {epoch} | Train={train_loss:.4f} | Val={val_loss:.4f}")
            history.append([epoch, train_loss, val_loss])
            torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, "shared.pt"))

    if rank == 0:
        df = pd.DataFrame(history, columns=["Epoch", "TrainLoss", "ValLoss"])
        df.to_csv(os.path.join(Config.SAVE_DIR, "losses.csv"), index=False)

        plt.figure(figsize=(8, 5))
        plt.plot(df["TrainLoss"], label="Train")
        plt.plot(df["ValLoss"], label="Val")
        plt.legend()
        plt.savefig(os.path.join(Config.SAVE_DIR, "losses.png"))

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
