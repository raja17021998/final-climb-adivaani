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
    CSV_FILE = os.path.join(DATASET_DIR, "Hin_Gon.csv")
    ROOT_DIR = "/home/user/Desktop/Shashwat/final-climb/baseline-funs"
    SPM_PATH = os.path.join(ROOT_DIR, "joint_spm.model")
    MODEL_PATH = os.path.join(ROOT_DIR, "lstm-lstm", LANG_NAME, "best_model.pt")
    VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS = 8000, 256, 512, 2

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

def evaluate_independent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=Config.SPM_PATH)
    
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    encoder, decoder = Encoder().to(device), Decoder().to(device)
    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    
    # Strip DDP 'module.' prefix
    encoder.load_state_dict({k.replace('encoder.', '').replace('module.', ''): v for k, v in state_dict.items() if 'encoder' in k})
    decoder.load_state_dict({k.replace('decoder.', '').replace('module.', ''): v for k, v in state_dict.items() if 'decoder' in k})
    encoder.eval(); decoder.eval()

    df = pd.read_csv(Config.CSV_FILE)
    df= df[:1000]
    test_df = df.drop(df.sample(frac=0.8, random_state=42).index).sample(frac=0.5, random_state=42)
    
    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Eval {Config.LANG_NAME}"):
        hin, tribal = str(row.iloc[0]), str(row.iloc[1])
        src = torch.LongTensor(sp.encode(hin, add_bos=True, add_eos=True)).unsqueeze(1).to(device)
        
        with torch.no_grad():
            h, c = encoder(src)
            preds = [1]
            for _ in range(50):
                out, h, c = decoder(torch.LongTensor([preds[-1]]).to(device), h, c)
                idx = out.argmax(1).item()
                preds.append(idx)
                if idx == 2: break
        
        pred_sent = sp.decode(preds)
        b_score = bleu_metric.compute(predictions=[pred_sent], references=[[tribal]], smooth_method="exp")['score']
        c_score = chrf_metric.compute(predictions=[pred_sent], references=[[tribal]])['score']
        results.append([hin, tribal, pred_sent, b_score, c_score])

    res_df = pd.DataFrame(results, columns=['Actual Hindi', f'Actual {Config.LANG_NAME}', f'Predicted {Config.LANG_NAME}', 'Bleu', 'Chrf++'])
    res_df.to_csv(os.path.join(os.path.dirname(Config.MODEL_PATH), "eval_results.csv"), index=False)
    print(f"\n{Config.LANG_NAME} -> Mean BLEU: {res_df['Bleu'].mean():.2f} | Mean Chrf++: {res_df['Chrf++'].mean():.2f}")

if __name__ == "__main__":
    evaluate_independent()