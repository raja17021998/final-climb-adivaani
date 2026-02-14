# train_ddp.py
import os
import torch
import random
import sentencepiece as spm
import pandas as pd
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import config as cfg
from text_protection import protect_text

LOSS_BASE_DIR = os.path.join(cfg.PLOTS_DIR)
LOG_BASE_DIR = os.path.join(cfg.LOGS_DIR)
EARLY_STOP_PATIENCE = 3

def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank
    return False, 0, 0

def load_special_tokens(sp):
    token_to_id = {}
    for tok in ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]:
        try:
            token_to_id[tok] = sp.piece_to_id(tok)
        except:
            token_to_id[tok] = None
    return token_to_id

class SeqDataset(Dataset):
    def __init__(self, src, tgt, sp, bos_id, eos_id):
        self.src = src
        self.tgt = tgt
        self.sp = sp
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __len__(self):
        return len(self.src)

    def encode(self, text, add_special=True):
        text, _ = protect_text(text)
        ids = self.sp.encode(text, out_type=int)
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def __getitem__(self, idx):
        src_ids = self.encode(self.src[idx])
        tgt_ids = self.encode(self.tgt[idx])
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def collate(batch, pad_id):
    src, tgt = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=pad_id)
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=pad_id)
    return src, tgt

class Encoder(nn.Module):
    def __init__(self, pad_id):
        super().__init__()
        p = cfg.SEQ2SEQ_PARAMS
        self.hidden_size = p["hidden_dim"]
        self.num_layers = p["num_layers"]
        self.embed = nn.Embedding(p["vocab_size"], p["embedding_dim"], padding_idx=pad_id)
        self.lstm = nn.LSTM(
            p["embedding_dim"],
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=p["dropout"],
        )

    def forward(self, x):
        x = self.embed(x)
        outputs, (h, c) = self.lstm(x)
        h = h.view(self.num_layers, 2, x.size(0), self.hidden_size)
        c = c.view(self.num_layers, 2, x.size(0), self.hidden_size)
        h = torch.cat((h[:, 0], h[:, 1]), dim=2)
        c = torch.cat((c[:, 0], c[:, 1]), dim=2)
        return outputs, (h, c)

class Decoder(nn.Module):
    def __init__(self, pad_id):
        super().__init__()
        p = cfg.SEQ2SEQ_PARAMS
        self.hidden_size = p["hidden_dim"] * 2
        self.num_layers = p["num_layers"]
        self.embed = nn.Embedding(p["vocab_size"], p["embedding_dim"], padding_idx=pad_id)
        self.lstm = nn.LSTM(
            p["embedding_dim"],
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=p["dropout"],
        )
        self.fc = nn.Linear(self.hidden_size, p["vocab_size"])

    def forward(self, x, hidden):
        x = self.embed(x.unsqueeze(1))
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out.squeeze(1))
        return out, hidden

class Seq2Seq(nn.Module):
    def __init__(self, pad_id, bos_id):
        super().__init__()
        self.encoder = Encoder(pad_id)
        self.decoder = Decoder(pad_id)
        self.bos_id = bos_id

    def forward(self, src, tgt, teacher_forcing_ratio):
        batch, tgt_len = tgt.size()
        vocab_size = cfg.SEQ2SEQ_PARAMS["vocab_size"]
        outputs = torch.zeros(batch, tgt_len, vocab_size, device=src.device)

        _, hidden = self.encoder(src)
        inp = torch.full((batch,), self.bos_id, device=src.device, dtype=torch.long)

        for t in range(1, tgt_len):
            out, hidden = self.decoder(inp, hidden)
            outputs[:, t] = out
            top = out.argmax(1)
            inp = tgt[:, t] if random.random() < teacher_forcing_ratio else top
        return outputs

def plot_loss(train_l, val_l, path):
    plt.figure()
    plt.plot(train_l)
    plt.plot(val_l)
    plt.legend(["train", "val"])
    plt.savefig(path)
    plt.close()

def get_train_file(lang):
    if lang == "bhili": return os.path.join(cfg.DATA_DIR, "Bhi_Hin_Mar_Guj_Eng.csv")
    if lang == "mundari": return os.path.join(cfg.DATA_DIR, "Mun_Hin_Eng.csv")
    if lang == "gondi": return os.path.join(cfg.DATA_DIR, "Gon_Hin_Eng.csv")
    if lang == "santali": return os.path.join(cfg.DATA_DIR, "San_Hin_Eng.csv")
    if lang == "kui": return os.path.join(cfg.DATA_DIR, "Kui_Hin_Eng.csv")
    if lang == "garo": return os.path.join(cfg.DATA_DIR, "Garo_Hin_Eng.csv")
    raise ValueError(lang)

def apply_direction_limit(df, lang, direction):
    limit = cfg.DIRECTION_DATA_LIMIT.get(lang, {}).get(direction, None)
    if limit is not None:
        df = df.head(limit)
    return df

def train_direction(lang, direction, sp, pad_id, bos_id, eos_id, distributed, rank, device):
    src_lang, tgt_lang = direction.split("_")
    train_file = get_train_file(lang)

    df = pd.read_csv(train_file)
    src_col = cfg.LANGUAGE_COLUMN_MAP[lang][src_lang]
    tgt_col = cfg.LANGUAGE_COLUMN_MAP[lang][tgt_lang]
    df = df[[src_col, tgt_col]].dropna()

    df = apply_direction_limit(df, lang, direction)

    if cfg.DEBUG_MODE:
        limit = cfg.DIRECTION_DATA_LIMIT.get(lang, {}).get(direction, None)
        if limit is not None:
            df = df.head(limit)
    else:
        limit = cfg.DIRECTION_DATA_LIMIT.get(lang, {}).get(direction, None)
        if limit is not None:
            df = df.head(limit)


    train_df, val_df = train_test_split(df, test_size=cfg.VAL_SPLIT)

    train_ds = SeqDataset(train_df[src_col].tolist(), train_df[tgt_col].tolist(), sp, bos_id, eos_id)
    val_ds = SeqDataset(val_df[src_col].tolist(), val_df[tgt_col].tolist(), sp, bos_id, eos_id)

    train_sampler = DistributedSampler(train_ds) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.TRAINING_PARAMS["batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=lambda b: collate(b, pad_id),
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.TRAINING_PARAMS["batch_size"],
        sampler=val_sampler,
        shuffle=False,
        collate_fn=lambda b: collate(b, pad_id),
        num_workers=2,
        pin_memory=True,
    )

    model = Seq2Seq(pad_id, bos_id).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if distributed:
        model = DDP(model, device_ids=[device])

    opt = optim.Adam(model.parameters(), lr=cfg.TRAINING_PARAMS["learning_rate"])
    crit = nn.CrossEntropyLoss(ignore_index=pad_id)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0

    log_dir = os.path.join(LOG_BASE_DIR, lang, direction)
    loss_dir = os.path.join(LOSS_BASE_DIR, lang, direction)
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(loss_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")

    for epoch in range(cfg.TRAINING_PARAMS["num_epochs"]):
        if distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        tl = 0
        train_iter = train_loader if rank != 0 else tqdm(train_loader, desc=f"{lang}-{direction} | Epoch {epoch+1} Train")
        for src, tgt in train_iter:
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad()
            out = model(src, tgt, cfg.SEQ2SEQ_PARAMS["teacher_forcing_ratio"])
            loss = crit(out[:, 1:].reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            opt.step()
            tl += loss.item()
        tl /= len(train_loader)
        train_losses.append(tl)

        model.eval()
        vl = 0
        val_iter = val_loader if rank != 0 else tqdm(val_loader, desc=f"{lang}-{direction} | Epoch {epoch+1} Val")
        with torch.no_grad():
            for src, tgt in val_iter:
                src, tgt = src.to(device), tgt.to(device)
                out = model(src, tgt, 0)
                loss = crit(out[:, 1:].reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
                vl += loss.item()
        vl /= len(val_loader)

        if distributed:
            t = torch.tensor([vl], device=device)
            dist.all_reduce(t)
            vl = t.item() / dist.get_world_size()

        val_losses.append(vl)

        if rank == 0:
            log_line = f"Epoch {epoch+1} | Train Loss: {tl:.4f} | Val Loss: {vl:.4f}"
            print(f"[{lang} | {direction}] {log_line}")
            with open(log_file, "a") as f:
                f.write(log_line + "\n")

            if vl < best_val_loss:
                best_val_loss = vl
                patience_counter = 0
                save_dir = os.path.join(cfg.MODEL_SAVE_DIR, lang, direction)
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered for {lang}-{direction}")
                break

    if rank == 0:
        loss_path = os.path.join(loss_dir, "loss.jpg")
        plot_loss(train_losses, val_losses, loss_path)

def main():
    distributed, rank, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    sp = spm.SentencePieceProcessor(model_file=cfg.TOKENIZER_MODEL)
    special_tokens = load_special_tokens(sp)

    pad_id = special_tokens.get("<pad>") or 0
    bos_id = special_tokens.get("<s>") or 1
    eos_id = special_tokens.get("</s>") or 2

    for lang in cfg.TRIBAL_LANGS:
        for direction, flag in cfg.DIRECTION_CONFIG[lang].items():
            if not flag:
                continue
            train_direction(lang, direction, sp, pad_id, bos_id, eos_id, distributed, rank, device)
            if distributed:
                dist.barrier()

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
