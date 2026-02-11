import torch.nn as nn

class MLMHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_size),
            nn.Linear(cfg.hidden_size, cfg.vocab_size),
        )

    def forward(self, x):
        return self.net(x)
