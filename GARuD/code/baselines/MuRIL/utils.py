import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling


class BhiliDataset(Dataset):
    """
    Expects lists of Hindi and Bhili sentences.
    """
    def __init__(self, bhili_texts):
        self.bh = bhili_texts

    def __len__(self):  
        return len(self.bh)

    def __getitem__(self, idx):
        return {"bh": self.bh[idx]}


class DistillCollatorCustom:
    """
    Encodes Bhili with student tokenizer and applies MLM masking via DataCollator.
    """
    def __init__(self, student_tok, max_len=128, mlm_prob=0.15, device="cpu"):
        self.student_tok = student_tok
        self.max_len = max_len
        self.mlm_prob = mlm_prob
        self.device = device
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=student_tok,
            mlm=True,
            mlm_probability=mlm_prob,
            return_tensors="pt"
        )

    def __call__(self, batch):
        # Bhili (student) with MLM masking
        bh_sentences = [item["bh"] for item in batch]
        bh_enc = self.student_tok(
            bh_sentences,
            padding='longest',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        # Apply MLM masking
        bh_inputs = [
            {
                "input_ids": bh_enc["input_ids"][i],  # Shape: [seq_len]
                "attention_mask": bh_enc["attention_mask"][i]  # Shape: [seq_len]
            }
            for i in range(bh_enc["input_ids"].shape[0])
        ]
        # Apply MLM masking
        bh_mlm = self.mlm_collator(bh_inputs)
        
        return {
            "bh_input_ids": bh_mlm["input_ids"],
            "bh_attention_mask": bh_mlm["attention_mask"],
            "bh_labels": bh_mlm["labels"]
        }
