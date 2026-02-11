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

# Silent warnings and HF logging
warnings.filterwarnings("ignore")
logging.getLogger("sacrebleu").setLevel(logging.ERROR)

class Config:
    LANG_NAME = "Mundari"
    DATASET_DIR = "/home/user/Desktop/Shashwat/final-climb/datasets"
    CSV_FILE = os.path.join(DATASET_DIR, "Hin_Mun.csv")
    ROOT_DIR = "/home/user/Desktop/Shashwat/final-climb/baseline-funs"
    SPM_PATH = os.path.join(ROOT_DIR, "joint_spm.model")
    # Using the attn-specific save directory
    MODEL_PATH = os.path.join(ROOT_DIR, "lstm-attn", LANG_NAME, "best_model.pt")
    VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS = 8000, 256, 512, 1 
    DROPOUT = 0.5

# --- ARCHITECTURE (Exactly same as Training) ---

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
        h = torch.tanh(self.fc_h(torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)))
        c = torch.tanh(self.fc_c(torch.cat((c[-2,:,:], c[-1,:,:]), dim=1)))
        return outputs, (h.unsqueeze(0), c.unsqueeze(0))

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear((Config.HID_DIM * 2) + Config.HID_DIM, Config.HID_DIM)
        self.v = nn.Linear(Config.HID_DIM, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(src_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
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

# --- EVALUATION LOGIC ---

def evaluate_attention():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=Config.SPM_PATH)
    
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    # Initialize model
    attn = Attention()
    encoder = Encoder().to(device)
    decoder = Decoder(attn).to(device)
    
    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    
    # Strip DDP 'module.' prefix if present
    encoder.load_state_dict({k.replace('encoder.', '').replace('module.', ''): v for k, v in state_dict.items() if 'encoder' in k})
    decoder.load_state_dict({k.replace('decoder.', '').replace('module.', ''): v for k, v in state_dict.items() if 'decoder' in k})
    encoder.eval(); decoder.eval()

    df = pd.read_csv(Config.CSV_FILE)
    df = df[:1000] # Evaluatng on sample slice as in your request
    test_df = df.drop(df.sample(frac=0.8, random_state=42).index).sample(frac=0.5, random_state=42)
    
    

    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Eval {Config.LANG_NAME} (Attn)"):
        hin, tribal = str(row.iloc[0]), str(row.iloc[1])
        src = torch.LongTensor(sp.encode(hin, add_bos=True, add_eos=True)).unsqueeze(1).to(device)
        
        with torch.no_grad():
            encoder_outputs, (h, c) = encoder(src)
            preds = [1] # BOS
            for _ in range(50):
                input_token = torch.LongTensor([preds[-1]]).to(device)
                out, h, c = decoder(input_token, h, c, encoder_outputs)
                idx = out.argmax(1).item()
                preds.append(idx)
                if idx == 2: break # EOS
        
        pred_sent = sp.decode(preds)
        
        # SacreBLEU compute fix for Hugging Face evaluate
        b_score = bleu_metric.compute(predictions=[pred_sent], references=[[tribal]], smooth_method="exp")['score']
        c_score = chrf_metric.compute(predictions=[pred_sent], references=[[tribal]])['score']
        results.append([hin, tribal, pred_sent, b_score, c_score])

    res_df = pd.DataFrame(results, columns=['Actual Hindi', f'Actual {Config.LANG_NAME}', f'Predicted {Config.LANG_NAME}', 'Bleu', 'Chrf++'])
    res_df.to_csv(os.path.join(os.path.dirname(Config.MODEL_PATH), "attn_eval_results.csv"), index=False)
    print(f"\n{Config.LANG_NAME} (Attn) -> Mean BLEU: {res_df['Bleu'].mean():.2f} | Mean Chrf++: {res_df['Chrf++'].mean():.2f}")

if __name__ == "__main__":
    evaluate_attention()