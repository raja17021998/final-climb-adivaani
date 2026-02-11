from dataclasses import dataclass, asdict
import json

@dataclass
class BertConfig:
    vocab_size: int
    max_len: int = 128

    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 8
    intermediate_size: int = 3072

    dropout: float = 0.1
    mlm_prob: float = 0.15

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
