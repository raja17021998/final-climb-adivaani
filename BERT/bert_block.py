import torch.nn as nn
from attention import MultiHeadAttention

class BertBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(cfg)
        self.ln1 = nn.LayerNorm(cfg.hidden_size)
        self.dropout1 = nn.Dropout(cfg.dropout)

        self.ff = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size),
        )
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, x, pad_mask=None):
        x = x + self.dropout1(self.attn(self.ln1(x), pad_mask))
        x = x + self.dropout2(self.ff(self.ln2(x)))
        return x
