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

class Config:
    LANG_NAME = "Mundari"
    DATASET_DIR = "/home/user/Desktop/Shashwat/final-climb/datasets"
    CSV_FILE = os.path.join(DATASET_DIR, "Hin_Mun.csv")
    ROOT_DIR = "/home/user/Desktop/Shashwat/final-climb/baseline-funs"
    SPM_PATH = os.path.join(ROOT_DIR, "joint_spm.model")
    # Project directory for Attention model
    SAVE_DIR = os.path.join(ROOT_DIR, "lstm-attn", LANG_NAME)
    
    VOCAB_SIZE = 8000
    EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 1  # Standard for Bahdanau Attention
    DROPOUT = 0.5
    LR = 0.001
    EPOCHS = 50
    PATIENCE = 5
    BATCH_SIZE = 32

os.makedirs(Config.SAVE_DIR, exist_ok=True)

class NMTDataset(Dataset):
    def __init__(self, df, sp_path):
        self.sp = spm.SentencePieceProcessor(model_file=sp_path)
        self.src = [self.sp.encode(str(x), out_type=int, add_bos=True, add_eos=True) for x in df.iloc[:,0]]
        self.trg = [self.sp.encode(str(x), out_type=int, add_bos=True, add_eos=True) for x in df.iloc[:,1]]
    def __len__(self): return len(self.src)
    def __getitem__(self, i): return torch.tensor(self.src[i]), torch.tensor(self.trg[i])

def collate_fn(batch):
    src, trg = zip(*batch)
    return nn.utils.rnn.pad_sequence(src, padding_value=0), nn.utils.rnn.pad_sequence(trg, padding_value=0)

# --- BAHDANAU ATTENTION COMPONENTS ---

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.EMB_DIM)
        self.rnn = nn.LSTM(Config.EMB_DIM, Config.HID_DIM, Config.N_LAYERS, dropout=Config.DROPOUT, bidirectional=True)
        self.fc_h = nn.Linear(Config.HID_DIM * 2, Config.HID_DIM)
        self.fc_c = nn.Linear(Config.HID_DIM * 2, Config.HID_DIM)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (h, c) = self.rnn(embedded)
        # outputs: [src_len, batch_size, hid_dim * 2]
        # Bridge bidirectional states to decoder initial state
        h = torch.tanh(self.fc_h(torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)))
        c = torch.tanh(self.fc_c(torch.cat((c[-2,:,:], c[-1,:,:]), dim=1)))
        return outputs, (h.unsqueeze(0), c.unsqueeze(0))

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear((Config.HID_DIM * 2) + Config.HID_DIM, Config.HID_DIM)
        self.v = nn.Linear(Config.HID_DIM, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch_size, hid_dim]
        # encoder_outputs: [src_len, batch_size, hid_dim * 2]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(src_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        # score = v * tanh(W * [h; enc_out])
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2) 
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.EMB_DIM)
        self.rnn = nn.LSTM((Config.HID_DIM * 2) + Config.EMB_DIM, Config.HID_DIM, Config.N_LAYERS)
        self.fc_out = nn.Linear((Config.HID_DIM * 2) + Config.HID_DIM + Config.EMB_DIM, Config.VOCAB_SIZE)
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, input, h, c, encoder_outputs):
        embedded = self.dropout(self.embedding(input.unsqueeze(0)))
        a = self.attention(h, encoder_outputs).unsqueeze(1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        weighted = torch.bmm(a, encoder_outputs).transpose(0, 1)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (h, c) = self.rnn(rnn_input, (h, c))
        prediction = self.fc_out(torch.cat((output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)), dim=1))
        return prediction, h, c

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder, self.decoder, self.device = encoder, decoder, device

    def forward(self, src, trg, tf_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        outputs = torch.zeros(trg_len, batch_size, Config.VOCAB_SIZE).to(self.device)
        encoder_outputs, (h, c) = self.encoder(src)
        input = trg[0,:]
        for t in range(1, trg_len):
            out, h, c = self.decoder(input, h, c, encoder_outputs)
            outputs[t] = out
            input = trg[t] if np.random.random() < tf_ratio else out.argmax(1)
        return outputs

# --- TRAINING ENGINE ---

def main():
    world_size = torch.cuda.device_count()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(Config.CSV_FILE)[:10000]
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index).sample(frac=0.5, random_state=42)
    
    # Correctly defining train_ds here
    train_ds = NMTDataset(train_df, Config.SPM_PATH)
    val_ds = NMTDataset(val_df, Config.SPM_PATH)
    
    sampler = DistributedSampler(train_ds) if world_size > 1 else None
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, sampler=sampler, shuffle=(sampler is None), collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn)

    model = Seq2Seq(Encoder(), Decoder(Attention()), device).to(device)
    if world_size > 1: model = DDP(model, device_ids=[rank])
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    best_loss, counter = float('inf'), 0
    t_losses, v_losses = [], []

    

    for epoch in range(Config.EPOCHS):
        if sampler: sampler.set_epoch(epoch)
        model.train()
        e_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", disable=(rank != 0))
        for src, trg in train_bar:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            out = model(src, trg)
            loss = criterion(out[1:].view(-1, Config.VOCAB_SIZE), trg[1:].view(-1))
            loss.backward()
            optimizer.step()
            e_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for src, trg in val_loader:
                src, trg = src.to(device), trg.to(device)
                out = model(src, trg, tf_ratio=0)
                v_loss += criterion(out[1:].view(-1, Config.VOCAB_SIZE), trg[1:].view(-1)).item()
        
        if rank == 0:
            avg_t, avg_v = e_loss/len(train_loader), v_loss/len(val_loader)
            t_losses.append(avg_t); v_losses.append(avg_v)
            print(f"\n>>> Epoch {epoch}: Train {avg_t:.4f} | Val {avg_v:.4f}")
            if avg_v < best_loss:
                best_loss, counter = avg_v, 0
                torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, "best_model.pt"))
            else:
                counter += 1
                if counter >= Config.PATIENCE: break

    if rank == 0:
        plt.plot(t_losses, label='Train'); plt.plot(v_losses, label='Val')
        plt.legend(); plt.savefig(os.path.join(Config.SAVE_DIR, "loss.png"))

if __name__ == "__main__": main()