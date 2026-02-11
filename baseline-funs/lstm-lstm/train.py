import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
import sentencepiece as spm
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm  # Progress bar ke liye

warnings.filterwarnings("ignore")

class Config:
    LANG_NAME = "Mundari"
    DATASET_DIR = "/home/user/Desktop/Shashwat/final-climb/datasets"
    CSV_FILE = os.path.join(DATASET_DIR, "Hin_Mun.csv")
    ROOT_DIR = "/home/user/Desktop/Shashwat/final-climb/baseline-funs"
    SPM_PATH = os.path.join(ROOT_DIR, "joint_spm.model")
    SAVE_DIR = os.path.join(ROOT_DIR, "lstm-lstm", LANG_NAME)
    
    VOCAB_SIZE = 8000
    EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
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

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.EMB_DIM)
        self.rnn = nn.LSTM(Config.EMB_DIM, Config.HID_DIM, Config.N_LAYERS, dropout=Config.DROPOUT, bidirectional=True)
        self.fc_h = nn.Linear(Config.HID_DIM * 2, Config.HID_DIM * Config.N_LAYERS)
        self.fc_c = nn.Linear(Config.HID_DIM * 2, Config.HID_DIM * Config.N_LAYERS)

    def forward(self, src):
        _, (h, c) = self.rnn(self.embedding(src))
        h_combined = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)
        c_combined = torch.cat((c[-2,:,:], c[-1,:,:]), dim=1)
        h = torch.tanh(self.fc_h(h_combined)).view(Config.N_LAYERS, -1, Config.HID_DIM)
        c = torch.tanh(self.fc_c(c_combined)).view(Config.N_LAYERS, -1, Config.HID_DIM)
        return h, c

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.EMB_DIM)
        self.rnn = nn.LSTM(Config.EMB_DIM, Config.HID_DIM, Config.N_LAYERS, dropout=Config.DROPOUT)
        self.fc_out = nn.Linear(Config.HID_DIM, Config.VOCAB_SIZE)
    def forward(self, input, h, c):
        output, (h, c) = self.rnn(self.embedding(input.unsqueeze(0)), (h, c))
        return self.fc_out(output.squeeze(0)), h, c

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder, self.decoder, self.device = encoder, decoder, device
    def forward(self, src, trg, tf_ratio=0.5):
        trg_len, batch_size = trg.shape
        outputs = torch.zeros(trg_len, batch_size, Config.VOCAB_SIZE).to(self.device)
        h, c = self.encoder(src)
        input = trg[0,:]
        for t in range(1, trg_len):
            out, h, c = self.decoder(input, h, c)
            outputs[t] = out
            input = trg[t] if np.random.random() < tf_ratio else out.argmax(1)
        return outputs

def main():
    world_size = torch.cuda.device_count()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1: 
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Dataset Loading
    df = pd.read_csv(Config.CSV_FILE)
    df= df[:10000]
    train_df = df.sample(frac=0.8, random_state=42)
    temp_df = df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=42)
    
    train_ds = NMTDataset(train_df, Config.SPM_PATH)
    val_ds = NMTDataset(val_df, Config.SPM_PATH)
    
    sampler = DistributedSampler(train_ds) if world_size > 1 else None
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, sampler=sampler, shuffle=(sampler is None), collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn)

    model = Seq2Seq(Encoder(), Decoder(), device).to(device)
    if world_size > 1: 
        model = DDP(model, device_ids=[rank])
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    best_loss, counter = float('inf'), 0
    t_losses, v_losses = [], []

    for epoch in range(Config.EPOCHS):
        if sampler: sampler.set_epoch(epoch)
        
        # --- TRAINING STRIDE ---
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

        # --- VALIDATION STRIDE ---
        model.eval()
        v_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", disable=(rank != 0))
        
        with torch.no_grad():
            for src, trg in val_bar:
                src, trg = src.to(device), trg.to(device)
                out = model(src, trg, tf_ratio=0)
                loss = criterion(out[1:].view(-1, Config.VOCAB_SIZE), trg[1:].view(-1))
                v_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())
        
        if rank == 0:
            avg_t = e_loss / len(train_loader)
            avg_v = v_loss / len(val_loader)
            t_losses.append(avg_t)
            v_losses.append(avg_v)
            
            print(f"\n>>> Epoch {epoch} Summary: Train Loss: {avg_t:.4f} | Val Loss: {avg_v:.4f}")
            
            if avg_v < best_loss:
                best_loss, counter = avg_v, 0
                torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, "best_model.pt"))
                print(f"[*] Model saved at {Config.SAVE_DIR}")
            else:
                counter += 1
                print(f"[!] Patience counter: {counter}/{Config.PATIENCE}")
                if counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

    if rank == 0:
        plt.figure(figsize=(10,5))
        plt.plot(t_losses, label='Train Loss')
        plt.plot(v_losses, label='Val Loss')
        plt.title(f"Loss Curves for {Config.LANG_NAME}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(Config.SAVE_DIR, "loss.png"))
        print(f"Loss plot saved in {Config.SAVE_DIR}")

if __name__ == "__main__":
    main()