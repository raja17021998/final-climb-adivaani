import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import sentencepiece as spm
import evaluate
from tqdm import tqdm
import warnings
import logging

# Warnings aur Sacrebleu ke spam ko silent karne ke liye
warnings.filterwarnings("ignore")
logging.getLogger("sacrebleu").setLevel(logging.ERROR)

class Config:
    ROOT_DIR = "/home/user/Desktop/Shashwat/final-climb/baseline-funs"
    DATASET_DIR = "/home/user/Desktop/Shashwat/final-climb/datasets"
    SPM_PATH = os.path.join(ROOT_DIR, "joint_spm.model")
    MODEL_PATH = os.path.join(ROOT_DIR, "lstm-lstm", "Shared_Model", "shared_best.pt")
    
    LANGS = ["bhili", "gondi", "kui", "mundari"]
    CSV_FILES = {
        "bhili": "Hin_Bhi_Mar_Guj.csv", 
        "gondi": "Hin_Gon.csv", 
        "kui": "Hin_Kui.csv", 
        "mundari": "Hin_Mun.csv"
    }
    
    # --- YAHAN SELECT KARO KITNE ROWS CHAHIYE ---
    # Example: {"bhili": 100, "gondi": 500, "kui": None (for all)}
    EVAL_LIMITS = {
        "bhili": 1000, 
        "gondi": 1000, 
        "kui": 1000, 
        "mundari": 1000
    }
    
    VOCAB_SIZE = 8000
    EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2

# --- Model Components ---

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.EMB_DIM)
        self.rnn = nn.LSTM(Config.EMB_DIM, Config.HID_DIM, Config.N_LAYERS, bidirectional=True)
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
        self.rnn = nn.LSTM(Config.EMB_DIM, Config.HID_DIM, Config.N_LAYERS)
        self.fc_out = nn.Linear(Config.HID_DIM, Config.VOCAB_SIZE)

    def forward(self, input, h, c):
        output, (h, c) = self.rnn(self.embedding(input.unsqueeze(0)), (h, c))
        return self.fc_out(output.squeeze(0)), h, c

# --- Evaluation Engine ---

def evaluate_shared():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=Config.SPM_PATH)
    
    # Load HF Metrics
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    # Initialize Architecture
    encoder = Encoder().to(device)
    decoders = nn.ModuleDict({l: Decoder().to(device) for l in Config.LANGS})
    
    # Load Checkpoint
    if not os.path.exists(Config.MODEL_PATH):
        print(f"Error: Model file not found at {Config.MODEL_PATH}")
        return

    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    
    # Clean keys (handling DDP 'module.' prefix)
    encoder.load_state_dict({k.replace('encoder.', '').replace('module.', ''): v for k, v in state_dict.items() if 'encoder' in k})
    for l in Config.LANGS:
        dec_weights = {k.replace(f'decoders.{l}.', '').replace('module.', ''): v for k, v in state_dict.items() if f'decoders.{l}' in k}
        decoders[l].load_state_dict(dec_weights)
    
    encoder.eval()
    decoders.eval()

    

    for l_name in Config.LANGS:
        csv_path = os.path.join(Config.DATASET_DIR, Config.CSV_FILES[l_name])
        if not os.path.exists(csv_path):
            print(f"Skipping {l_name}: CSV not found at {csv_path}")
            continue

        # Selection logic based on Config.EVAL_LIMITS
        df = pd.read_csv(csv_path)
        limit = Config.EVAL_LIMITS.get(l_name)
        if limit is not None:
            test_df = df.head(limit)
            print(f"\nEvaluating {l_name.upper()} (Limited to {limit} rows)...")
        else:
            test_df = df
            print(f"\nEvaluating {l_name.upper()} (Full CSV)...")
        
        results = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Inference"):
            hin_txt = str(row.iloc[0])
            target_txt = str(row.iloc[1])
            
            # Preprocess
            src_tokens = sp.encode(hin_txt, out_type=int, add_bos=True, add_eos=True)
            src_tensor = torch.LongTensor(src_tokens).unsqueeze(1).to(device)
            
            with torch.no_grad():
                h, c = encoder(src_tensor)
                preds = [1] # BOS token ID
                
                # Greedy Decoding
                for _ in range(60): # Max length
                    input_token = torch.LongTensor([preds[-1]]).to(device)
                    out, h, c = decoders[l_name](input_token, h, c)
                    next_token = out.argmax(1).item()
                    preds.append(next_token)
                    if next_token == 2: # EOS token ID
                        break
            
            pred_txt = sp.decode(preds)
            
            # Metrics Calculation (using the fix for Sacrebleu smooth_method)
            b_score = bleu_metric.compute(predictions=[pred_txt], references=[[target_txt]], smooth_method="exp")['score']
            c_score = chrf_metric.compute(predictions=[pred_txt], references=[[target_txt]])['score']
            
            results.append([hin_txt, target_txt, pred_txt, b_score, c_score])

        # Save results to CSV
        res_df = pd.DataFrame(results, columns=['Actual Hindi', f'Actual {l_name}', f'Predicted {l_name}', 'Bleu', 'Chrf++'])
        out_csv = os.path.join(os.path.dirname(Config.MODEL_PATH), f"shared_eval_{l_name}.csv")
        res_df.to_csv(out_csv, index=False)
        
        print(f"Results saved to: {out_csv}")
        print(f"Mean BLEU: {res_df['Bleu'].mean():.2f} | Mean Chrf++: {res_df['Chrf++'].mean():.2f}")

if __name__ == "__main__":
    evaluate_shared()