import torch.nn as nn
from bert_block import BertBlock

class BertModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([BertBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.LayerNorm(cfg.hidden_size)

    def forward(self, ids):
        x = self.emb(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
