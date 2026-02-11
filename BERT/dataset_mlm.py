import torch
from torch.utils.data import Dataset
import random

class MLMDataset(Dataset):
    def __init__(self, sequences, cfg, cls, sep, mask, pad):
        self.data = sequences
        self.cfg = cfg
        self.cls, self.sep = cls, sep
        self.mask, self.pad = mask, pad

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx][: self.cfg.max_len - 2]
        ids = [self.cls] + ids + [self.sep]

        labels = [-100] * len(ids)

        for i in range(1, len(ids) - 1):
            if random.random() < self.cfg.mlm_prob:
                labels[i] = ids[i]
                r = random.random()
                if r < 0.8:
                    ids[i] = self.mask
                elif r < 0.9:
                    ids[i] = random.randint(0, self.cfg.vocab_size - 1)

        pad_len = self.cfg.max_len - len(ids)
        ids += [self.pad] * pad_len
        labels += [-100] * pad_len

        return torch.tensor(ids), torch.tensor(labels)
