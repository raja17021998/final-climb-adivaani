import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import sentencepiece as spm
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ===================== CONFIG =====================

class Config:
    ROOT_DIR = "/home/user/Desktop/Shashwat/final-climb/baseline-funs"
    DATASET_DIR = "/home/user/Desktop/Shashwat/final-climb/datasets"
    SPM_PATH = os.path.join(ROOT_DIR, "joint_spm.model")
    SAVE_DIR = os.path.join(ROOT_DIR, "lstm-attn", "Shared_Model")

    LANGS = ["bhili", "gondi", "kui", "mundari"]
    CSV_FILES = {
        "bhili": "Hin_Bhi_Mar_Guj.csv",
        "gondi": "Hin_Gon.csv",
        "kui": "Hin_Kui.csv",
        "mundari": "Hin_Mun.csv"
    }

    LIMITS = {l: 10000 for l in LANGS}

    VOCAB_SIZE = 8000
    EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 1
    DROPOUT = 0.5
    LR = 0.001
    EPOCHS = 5
    PATIENCE = 5
    BATCH_SIZE = 2

os.makedirs(Config.SAVE_DIR, exist_ok=True)
torch.cuda.empty_cache()

# ===================== DATA =====================

class MultiTaskDataset(Dataset):
    def __init__(self, df, lang_id, sp_path):
        self.sp = spm.SentencePieceProcessor(model_file=sp_path)
        self.lang_id = lang_id
        self.src = [self.sp.encode(str(x), out_type=int, add_bos=True, add_eos=True) for x in df.iloc[:, 0]]
        self.trg = [self.sp.encode(str(x), out_type=int, add_bos=True, add_eos=True) for x in df.iloc[:, 1]]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return torch.tensor(self.src[i]), torch.tensor(self.trg[i]), self.lang_id


def shared_collate(batch):
    src, trg, ids = zip(*batch)
    return (
        nn.utils.rnn.pad_sequence(src, padding_value=0),
        nn.utils.rnn.pad_sequence(trg, padding_value=0),
        torch.tensor(ids)
    )

# ===================== MODEL =====================

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.EMB_DIM)
        self.rnn = nn.LSTM(
            Config.EMB_DIM,
            Config.HID_DIM,
            Config.N_LAYERS,
            dropout=Config.DROPOUT,
            bidirectional=True
        )
        self.fc_h = nn.Linear(Config.HID_DIM * 2, Config.HID_DIM)
        self.fc_c = nn.Linear(Config.HID_DIM * 2, Config.HID_DIM)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (h, c) = self.rnn(embedded)

        h = torch.tanh(self.fc_h(torch.cat((h[-2], h[-1]), dim=1)))
        c = torch.tanh(self.fc_c(torch.cat((c[-2], c[-1]), dim=1)))

        return outputs, (h.unsqueeze(0), c.unsqueeze(0))


class SharedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(Config.HID_DIM * 3, Config.HID_DIM)
        self.v = nn.Linear(Config.HID_DIM, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(src_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return torch.softmax(attention, dim=1)


class TribalDecoder(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.EMB_DIM)
        self.rnn = nn.LSTM(
            Config.HID_DIM * 2 + Config.EMB_DIM,
            Config.HID_DIM,
            Config.N_LAYERS
        )
        self.fc_out = nn.Linear(
            Config.HID_DIM * 3 + Config.EMB_DIM,
            Config.VOCAB_SIZE
        )
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, input, h, c, encoder_outputs):
        embedded = self.dropout(self.embedding(input.unsqueeze(0)))

        a = self.attention(h, encoder_outputs).unsqueeze(1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        weighted = torch.bmm(a, encoder_outputs).transpose(0, 1)

        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (h, c) = self.rnn(rnn_input, (h, c))

        pred = self.fc_out(
            torch.cat(
                (output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)),
                dim=1
            )
        )
        return pred, h, c


class SharedSeq2Seq(nn.Module):
    def __init__(self, encoder, decoders, device):
        super().__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleDict(decoders)
        self.device = device

    def forward(self, src, trg, lang_name, criterion, tf_ratio=0.5):
        encoder_outputs, (h, c) = self.encoder(src)

        decoder = self.decoders[lang_name]
        input = trg[0]
        loss = 0

        for t in range(1, trg.shape[0]):
            out, h, c = decoder(input, h, c, encoder_outputs)
            loss += criterion(out, trg[t])
            input = trg[t] if np.random.random() < tf_ratio else out.argmax(1)

        return loss / (trg.shape[0] - 1)

# ===================== TRAIN =====================

def main():
    world_size = torch.cuda.device_count()
    rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    train_loaders, val_loaders, samplers = {}, {}, {}

    for idx, lang in enumerate(Config.LANGS):
        df = pd.read_csv(os.path.join(Config.DATASET_DIR, Config.CSV_FILES[lang]))[:Config.LIMITS[lang]]
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)

        train_ds = MultiTaskDataset(train_df, idx, Config.SPM_PATH)
        sampler = DistributedSampler(train_ds) if world_size > 1 else None

        train_loaders[lang] = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=shared_collate
        )

        val_loaders[lang] = DataLoader(
            MultiTaskDataset(val_df, idx, Config.SPM_PATH),
            batch_size=Config.BATCH_SIZE,
            collate_fn=shared_collate
        )

        samplers[lang] = sampler

    shared_attn = SharedAttention()
    encoder = Encoder()
    decoders = {l: TribalDecoder(shared_attn) for l in Config.LANGS}

    model = SharedSeq2Seq(encoder, decoders, device).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    history = {l: [] for l in Config.LANGS}
    best_v, patience = float("inf"), 0

    for epoch in range(Config.EPOCHS):
        model.train()
        for l in Config.LANGS:
            if samplers[l]:
                samplers[l].set_epoch(epoch)

        iters = {l: iter(train_loaders[l]) for l in Config.LANGS}
        max_batches = max(len(train_loaders[l]) for l in Config.LANGS)

        bar = tqdm(range(max_batches), desc=f"Epoch {epoch} [Shared Attn]", disable=(rank != 0))

        for _ in bar:
            for l in Config.LANGS:
                try:
                    src, trg, _ = next(iters[l])
                except StopIteration:
                    iters[l] = iter(train_loaders[l])
                    src, trg, _ = next(iters[l])

                src, trg = src.to(device), trg.to(device)

                optimizer.zero_grad()
                loss = model(src, trg, l, criterion)
                loss.backward()
                optimizer.step()

                bar.set_postfix(lang=l, loss=loss.item())

        # -------- VALIDATION --------
        model.eval()
        with torch.no_grad():
            for l, loader in val_loaders.items():
                total = 0
                for src, trg, _ in loader:
                    src, trg = src.to(device), trg.to(device)
                    total += model(src, trg, l, criterion, tf_ratio=0).item()
                history[l].append(total / len(loader))

        if rank == 0:
            mean_v = np.mean([history[l][-1] for l in Config.LANGS])
            print(f"\n>>> Epoch {epoch} Mean Val Loss: {mean_v:.4f}")

            if mean_v < best_v:
                best_v, patience = mean_v, 0
                torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, "shared_best.pt"))
            else:
                patience += 1
                if patience >= Config.PATIENCE:
                    break

    if rank == 0:
        plt.figure(figsize=(10, 6))
        for l in Config.LANGS:
            plt.plot(history[l], label=l)
        plt.legend()
        plt.savefig(os.path.join(Config.SAVE_DIR, "val_loss.png"))

if __name__ == "__main__":
    main()
