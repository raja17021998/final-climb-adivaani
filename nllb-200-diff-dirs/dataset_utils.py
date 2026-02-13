# dataset_utils.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from config import *
from text_protection import protect_text

DATASET_FILES = {
    "bhili": "Bhi_Hin_Mar_Guj_Eng.csv",
    "mundari": "Mun_Hin_Eng.csv",
    "gondi": "Gon_Hin_Eng.csv",
    "santali": "San_Hin_Eng.csv",
    "kui": "Kui_Hin_Eng.csv",
    "garo": "Garo_Hin_Eng.csv",
}

TEST_FILES = {
    "bhili": "test_Bhi_Hin_Mar_Guj_Eng.csv",
    "mundari": "test_Mun_Hin_Eng.csv",
    "gondi": "test_Gon_Hin_Eng.csv",
    "santali": "test_San_Hin_Eng.csv",
    "kui": "test_Kui_Hin_Eng.csv",
    "garo": "test_Garo_Hin_Eng.csv",
}

def get_dataset_path(tribal):
    return os.path.join(DATA_DIR, DATASET_FILES[tribal])

def get_test_dataset_path(tribal):
    return os.path.join(TEST_DATA_DIR, TEST_FILES[tribal])

class TranslationDataset(Dataset):
    def __init__(self, df, src_col, tgt_col, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = str(self.df.loc[idx, self.src_col])
        tgt = str(self.df.loc[idx, self.tgt_col])

        src_p, _ = protect_text(src)
        tgt_p, _ = protect_text(tgt)

        inputs = self.tokenizer(
            src_p,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            tgt_p,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"]

        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = labels.squeeze(0)
        return item

def load_split_dataset(tribal):
    df = pd.read_csv(get_dataset_path(tribal))
    if DEBUG_MODE:
        df = df.head(DEBUG_TRAIN_ROWS)
    train_df, val_df = train_test_split(df, test_size=VAL_SPLIT, random_state=42)
    return train_df, val_df
