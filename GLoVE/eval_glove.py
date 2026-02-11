# ============================================================
# Simple GLoVe evaluator (nearest neighbors)
# ============================================================

import torch
import numpy as np
from pathlib import Path

BASE = Path("/home/shashwat1/final-climb-shashwat-do-not-delete/GLoVe")


def eval_lang(lang: str, word: str, topk: int = 10):
    data = torch.load(BASE / lang / f"{lang.lower()}_glove.pt", weights_only=False)

    vocab = data["vocab"]
    id2w = data["id_to_word"]

    wi = data["weights"]["wi.weight"].cpu().numpy()
    wj = data["weights"]["wj.weight"].cpu().numpy()

    # Standard GLoVe word vectors
    emb = wi + wj
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)

    if word not in vocab:
        print(f"[OOV] {word}")
        return

    v = emb[vocab[word]]
    sims = emb @ v

    print(f"\nNearest neighbors for '{word}' ({lang})")
    for i in np.argsort(-sims)[:topk]:
        print(f"{id2w[i]:20s} {sims[i]:.4f}")


if __name__ == "__main__":
    eval_lang("Bhili", "पूर्ति")
