import torch
import torch.nn as nn

def rotate_half(x):
    # Splits the last dimension (head_dim) into two halves
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

class RoPE(nn.Module):
    def __init__(self, head_dim, max_len=2048, base=10000):
        super().__init__()
        # Calculate inverse frequencies in float32 for precision
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        
        # RoPE usually expects cos and sin to be duplicated for x1 and x2
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, :, None, :])
        self.register_buffer("sin", emb.sin()[None, :, None, :])

    def forward(self, x):
        # x shape: [Batch, Seq_len, Num_heads, Head_dim]
        seq_len = x.shape[1]
        cos = self.cos[:, :seq_len, :, :]
        sin = self.sin[:, :seq_len, :, :]
        return (x * cos) + (rotate_half(x) * sin)