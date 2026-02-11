import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    SAVE_DIR = os.path.join(ROOT_DIR, "RoFormer", "shared")

    LANGS = ["bhili", "gondi", "kui", "mundari"]
    CSV_FILES = {
        "bhili": "Hin_Bhi_Mar_Guj.csv",
        "gondi": "Hin_Gon.csv",
        "kui": "Hin_Kui.csv",
        "mundari": "Hin_Mun.csv"
    }

    TRAIN_LIMITS = {
        "bhili": 50,
        "gondi": 40,
        "kui": 30,
        "mundari": 50
    }

    VAL_SPLIT = 0.1
    TEMP = 0.7   # Temperature for multilingual sampling

    VOCAB_SIZE = 8000
    D_MODEL = 512
    N_HEADS = 8
    ENC_LAYERS = 6
    DEC_LAYERS = 6
    FF_DIM = 2048

    BATCH_SIZE = 32
    LR = 3e-4
    EPOCHS = 10
    PAD_ID = 0

# ================= DISTRIBUTED SETUP =================

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
        return rank, world_size, torch.device(f"cuda:{rank}"), True
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu"), False

# ================= DATASET & SAMPLING =================

class LangDataset(Dataset):
    def __init__(self, df, sp_path):
        self.df = df.reset_index(drop=True)
        self.sp = spm.SentencePieceProcessor(model_file=sp_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        src = self.sp.encode(str(self.df.iloc[i, 0]), add_bos=True, add_eos=True)
        tgt = self.sp.encode(str(self.df.iloc[i, 1]), add_bos=True, add_eos=True)
        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    src, tgt = zip(*batch)
    # Using batch_first=True to match nn.MultiheadAttention expectations
    src_padded = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=Config.PAD_ID)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=Config.PAD_ID)
    return src_padded, tgt_padded

def compute_sampling_probs(sizes, temperature):
    """
    Computes temperature-scaled sampling probabilities for multilingual data.
    """
    sizes = torch.tensor(sizes, dtype=torch.float)
    scaled = sizes.pow(1.0 / temperature)
    return scaled / scaled.sum()

def build_language_schedule(langs, probs, steps, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.multinomial(probs, steps, replacement=True, generator=g)
    return [langs[i] for i in idx.tolist()]

# ================= MODEL ARCHITECTURE =================

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

class RoAttention(nn.Module):
    def __init__(self, is_causal=False):
        super().__init__()
        self.h = Config.N_HEADS
        self.d = Config.D_MODEL // Config.N_HEADS
        self.qkv = nn.Linear(Config.D_MODEL, 3 * Config.D_MODEL)
        self.out = nn.Linear(Config.D_MODEL, Config.D_MODEL)
        self.rope = RotaryEmbedding(self.d)
        self.is_causal = is_causal

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.d).transpose(1, 2)
        k = k.view(B, T, self.h, self.d).transpose(1, 2)
        v = v.view(B, T, self.h, self.d).transpose(1, 2)

        offset = cache[0].size(2) if cache is not None else 0
        cos, sin = self.rope(T + offset, x.device)
        
        # Reshape RoPE embeddings for broadcasting across heads
        cos = cos[offset:offset+T].unsqueeze(0).unsqueeze(0)
        sin = sin[offset:offset+T].unsqueeze(0).unsqueeze(0)
        
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal and T > 1)
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.out(y), (k, v)



class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = RoAttention(is_causal=False)
        self.ff = nn.Sequential(
            nn.Linear(Config.D_MODEL, Config.FF_DIM),
            nn.GELU(),
            nn.Linear(Config.FF_DIM, Config.D_MODEL)
        )
        self.ln1 = nn.LayerNorm(Config.D_MODEL)
        self.ln2 = nn.LayerNorm(Config.D_MODEL)

    def forward(self, x):
        attn_out, _ = self.attn(self.ln1(x))
        x = x + attn_out
        return x + self.ff(self.ln2(x))

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = RoAttention(is_causal=True)
        self.cross_attn = nn.MultiheadAttention(Config.D_MODEL, Config.N_HEADS, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(Config.D_MODEL, Config.FF_DIM),
            nn.GELU(),
            nn.Linear(Config.FF_DIM, Config.D_MODEL)
        )
        self.ln1 = nn.LayerNorm(Config.D_MODEL)
        self.ln2 = nn.LayerNorm(Config.D_MODEL)
        self.ln3 = nn.LayerNorm(Config.D_MODEL)

    def forward(self, x, mem, cache=None):
        x_attn, new_cache = self.self_attn(self.ln1(x), cache=cache)
        x = x + x_attn
        # Important: Cross-attention between decoder hidden states (x) and encoder memory (mem)
        x_cross, _ = self.cross_attn(self.ln2(x), mem, mem)
        x = x + x_cross
        return x + self.ff(self.ln3(x)), new_cache

class SharedTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(Config.VOCAB_SIZE, Config.D_MODEL)
        self.encoder = nn.ModuleList([EncoderBlock() for _ in range(Config.ENC_LAYERS)])
        self.decoders = nn.ModuleDict({
            l: nn.ModuleList([DecoderBlock() for _ in range(Config.DEC_LAYERS)])
            for l in Config.LANGS
        })
        self.heads = nn.ModuleDict({
            l: nn.Linear(Config.D_MODEL, Config.VOCAB_SIZE)
            for l in Config.LANGS
        })

    def forward(self, src, tgt, lang, caches=None):
        # src/tgt are Batch-First (B, T)
        enc = self.emb(src)
        for blk in self.encoder:
            enc = blk(enc)
        
        dec = self.emb(tgt)
        new_caches = []
        for i, blk in enumerate(self.decoders[lang]):
            c = caches[i] if caches else None
            dec, nc = blk(dec, enc, cache=c)
            new_caches.append(nc)
            
        return self.heads[lang](dec), new_caches

# ================= TRAINING LOOP =================

def main():
    rank, world_size, device, ddp = setup_distributed()
    if rank == 0:
        os.makedirs(Config.SAVE_DIR, exist_ok=True)

    model = SharedTransformer().to(device)
    if ddp:
        model = DDP(model, device_ids=[rank])

    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_ID)

    # ---------- Data Preparation ----------
    train_loaders, val_loaders, samplers, sizes = {}, {}, {}, []
    
    for lang in Config.LANGS:
        csv_path = os.path.join(Config.DATASET_DIR, Config.CSV_FILES[lang])
        df = pd.read_csv(csv_path)
        if Config.TRAIN_LIMITS.get(lang):
            df = df.head(Config.TRAIN_LIMITS[lang])

        split = int(len(df) * (1 - Config.VAL_SPLIT))
        train_df, val_df = df.iloc[:split], df.iloc[split:]

        train_ds = LangDataset(train_df, Config.SPM_PATH)
        val_ds = LangDataset(val_df, Config.SPM_PATH)

        sampler = DistributedSampler(train_ds) if ddp else None
        train_loaders[lang] = DataLoader(
            train_ds, Config.BATCH_SIZE, sampler=sampler, 
            shuffle=(sampler is None), collate_fn=collate_fn
        )
        val_loaders[lang] = DataLoader(val_ds, Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        samplers[lang] = sampler
        sizes.append(len(train_ds))

    probs = compute_sampling_probs(sizes, Config.TEMP)
    history = []

    # ---------- Training epochs ----------
    for ep in range(Config.EPOCHS):
        model.train()
        if ddp:
            for s in samplers.values(): s.set_epoch(ep)

        iters = {l: iter(train_loaders[l]) for l in Config.LANGS}
        max_batches = max(len(train_loaders[l]) for l in Config.LANGS)
        total_steps = max_batches * len(Config.LANGS)
        
        schedule = build_language_schedule(Config.LANGS, probs, total_steps, seed=ep + rank)
        
        epoch_loss = 0.0
        pbar = tqdm(schedule, desc=f"Epoch {ep+1}", disable=(rank != 0))
        
        for lang in pbar:
            try:
                src, tgt = next(iters[lang])
            except StopIteration:
                iters[lang] = iter(train_loaders[lang])
                src, tgt = next(iters[lang])

            src, tgt = src.to(device), tgt.to(device)
            
            # Causal training: Input is tgt sequence up to N-1, target is sequence from 1 to N
            optimizer.zero_grad()
            logits, _ = model(src, tgt[:, :-1], lang)
            
            loss = criterion(
                logits.reshape(-1, Config.VOCAB_SIZE),
                tgt[:, 1:].reshape(-1)
            )
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if rank == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lang": lang})

        avg_train_loss = epoch_loss / total_steps

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lang in Config.LANGS:
                for src, tgt in val_loaders[lang]:
                    src, tgt = src.to(device), tgt.to(device)
                    logits, _ = model(src, tgt[:, :-1], lang)
                    loss = criterion(
                        logits.reshape(-1, Config.VOCAB_SIZE),
                        tgt[:, 1:].reshape(-1)
                    )
                    val_loss += loss.item()
        
        avg_val_loss = val_loss / sum(len(val_loaders[l]) for l in Config.LANGS)

        if rank == 0:
            print(f"\n[SUMMARY] Epoch {ep+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            history.append([ep, avg_train_loss, avg_val_loss])
            
            # Saving model
            save_path = os.path.join(Config.SAVE_DIR, "shared.pt")
            torch.save(model.module.state_dict() if ddp else model.state_dict(), save_path)

    # ---------- Post-Training ----------
    if rank == 0:
        hist_df = pd.DataFrame(history, columns=["Epoch", "Train", "Val"])
        hist_df.to_csv(os.path.join(Config.SAVE_DIR, "loss_history.csv"), index=False)
        plt.figure(figsize=(10, 6))
        plt.plot(hist_df["Train"], label="Train Loss")
        plt.plot(hist_df["Val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(Config.SAVE_DIR, "loss_plot.png"))

    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()