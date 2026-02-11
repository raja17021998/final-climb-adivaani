import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, ConcatDataset
import sentencepiece as spm
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

class Config:
    ROOT_DIR = "/home/user/Desktop/Shashwat/final-climb/baseline-funs"
    DATASET_DIR = "/home/user/Desktop/Shashwat/final-climb/datasets"
    SPM_PATH = os.path.join(ROOT_DIR, "joint_spm.model")
    SAVE_DIR = os.path.join(ROOT_DIR, "lstm-lstm", "Shared_Model")
    
    LANGS = ["bhili", "gondi", "kui", "mundari"]
    CSV_FILES = {
        "bhili": "Hin_Bhi_Mar_Guj.csv",
        "gondi": "Hin_Gon.csv",
        "kui": "Hin_Kui.csv",
        "mundari": "Hin_Mun.csv"
    }
    # Debugging Slices - Set to None or larger values for full training
    LIMITS = {"bhili": 10000, "gondi": 10000, "kui": 10000, "mundari": 10000}
    
    VOCAB_SIZE = 8000
    EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    DROPOUT = 0.5
    LR = 0.001
    EPOCHS = 25
    PATIENCE = 5
    BATCH_SIZE = 128

os.makedirs(Config.SAVE_DIR, exist_ok=True)

class MultiTaskDataset(Dataset):
    def __init__(self, df, lang_id, sp_path):
        self.sp = spm.SentencePieceProcessor(model_file=sp_path)
        self.lang_id = lang_id
        self.src = [self.sp.encode(str(x), out_type=int, add_bos=True, add_eos=True) for x in df.iloc[:,0]]
        self.trg = [self.sp.encode(str(x), out_type=int, add_bos=True, add_eos=True) for x in df.iloc[:,1]]
    def __len__(self): return len(self.src)
    def __getitem__(self, i): return torch.tensor(self.src[i]), torch.tensor(self.trg[i]), self.lang_id

def shared_collate(batch):
    src, trg, ids = zip(*batch)
    return nn.utils.rnn.pad_sequence(src, padding_value=0), nn.utils.rnn.pad_sequence(trg, padding_value=0), torch.tensor(ids)

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

class SharedSeq2Seq(nn.Module):
    def __init__(self, encoder, decoders_dict, device):
        super().__init__()
        self.encoder, self.decoders, self.device = encoder, nn.ModuleDict(decoders_dict), device
    def forward(self, src, trg, lang_id, tf_ratio=0.5):
        h, c = self.encoder(src)
        trg_len, batch_size = trg.shape
        # Assumes batch has one language; taking first lang_id for decoder selection
        l_name = Config.LANGS[lang_id[0].item()]
        outputs = torch.zeros(trg_len, batch_size, Config.VOCAB_SIZE).to(self.device)
        input = trg[0,:]
        for t in range(1, trg_len):
            out, h, c = self.decoders[l_name](input, h, c)
            outputs[t] = out
            input = trg[t] if np.random.random() < tf_ratio else out.argmax(1)
        return outputs

def main():
    world_size = torch.cuda.device_count()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1: 
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    all_train_sets, val_loaders = [], {}
    for idx, l_name in enumerate(Config.LANGS):
        path = os.path.join(Config.DATASET_DIR, Config.CSV_FILES[l_name])
        df = pd.read_csv(path)[:Config.LIMITS[l_name]]
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index).sample(frac=0.5, random_state=42)
        
        all_train_sets.append(MultiTaskDataset(train_df, idx, Config.SPM_PATH))
        val_loaders[l_name] = DataLoader(MultiTaskDataset(val_df, idx, Config.SPM_PATH), batch_size=Config.BATCH_SIZE, collate_fn=shared_collate)

    combined_train = ConcatDataset(all_train_sets)
    sampler = DistributedSampler(combined_train) if world_size > 1 else None
    train_loader = DataLoader(combined_train, batch_size=Config.BATCH_SIZE, sampler=sampler, shuffle=(sampler is None), collate_fn=shared_collate)

    dec_dict = {l: Decoder() for l in Config.LANGS}
    model = SharedSeq2Seq(Encoder(), dec_dict, device).to(device)
    if world_size > 1: 
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    history = {l: [] for l in Config.LANGS}
    best_mean_v, patience = float('inf'), 0

    

    for epoch in range(Config.EPOCHS):
        if sampler: sampler.set_epoch(epoch)
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Shared Train]", disable=(rank != 0))
        for src, trg, l_ids in train_bar:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            out = model(src, trg, l_ids)
            loss = criterion(out[1:].view(-1, Config.VOCAB_SIZE), trg[1:].view(-1))
            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss.item())

        model.eval()
        current_v = []
        if rank == 0: print(f"\n--- Validation Epoch {epoch} ---")
        
        with torch.no_grad():
            for l_name, v_loader in val_loaders.items():
                v_l = 0
                for src, trg, l_ids in v_loader:
                    src, trg = src.to(device), trg.to(device)
                    out = model(src, trg, l_ids, tf_ratio=0)
                    v_l += criterion(out[1:].view(-1, Config.VOCAB_SIZE), trg[1:].view(-1)).item()
                avg_v = v_l/len(v_loader)
                history[l_name].append(avg_v)
                current_v.append(avg_v)
                if rank == 0: print(f" {l_name.capitalize()} Val Loss: {avg_v:.4f}")

        if rank == 0:
            m_v = np.mean(current_v)
            print(f">>> Mean Val Loss: {m_v:.4f}")
            if m_v < best_mean_v:
                best_mean_v, patience = m_v, 0
                torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, "shared_best.pt"))
                print(f"[*] Best Shared Model saved.")
            else:
                patience += 1
                print(f"[!] Patience: {patience}/{Config.PATIENCE}")
                if patience >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

    if rank == 0:
        plt.figure(figsize=(12,6))
        for l_name, losses in history.items():
            plt.plot(losses, label=f'{l_name.capitalize()}')
        plt.title("Multi-Task Validation Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(Config.SAVE_DIR, "shared_val_plot.png"))
        print(f"Multi-task loss plot saved in {Config.SAVE_DIR}")

if __name__ == "__main__": 
    main()