import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import RoPE

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h = cfg.num_heads
        self.d = cfg.hidden_size // cfg.num_heads

        assert self.d % 2 == 0, "RoPE head_dim must be even"

        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size)
        self.out = nn.Linear(cfg.hidden_size, cfg.hidden_size)

        self.rope = RoPE(self.d, max_len=cfg.max_len)

    def forward(self, x, pad_mask=None):
        B, T, C = x.shape

        qkv = self.qkv(x).view(B, T, 3, self.h, self.d)
        q, k, v = qkv.unbind(dim=2)

        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if pad_mask is not None:
            # [B, T] -> [B, 1, 1, T]
            attn_mask = pad_mask[:, None, None, :].to(torch.bool)
        else:
            attn_mask = None

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=False
        )

        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(attn)
