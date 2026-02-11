# ============================================================
# GLoVe Embedding Explorer
# ============================================================

import torch
import numpy as np
from pathlib import Path

BASE_DIR = Path("/home/shashwat1/final-climb-shashwat-do-not-delete/GLoVe")


class GloVeExplorer:
    def __init__(self, lang: str):
        self.lang = lang
        self.lang_dir = BASE_DIR / lang
        model_path = self.lang_dir / f"{lang.lower()}_glove.pt"

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        data = torch.load(model_path, weights_only=False)

        self.vocab = data["vocab"]
        self.id_to_word = data["id_to_word"]

        wi = data["weights"]["wi.weight"].cpu().numpy()
        wj = data["weights"]["wj.weight"].cpu().numpy()

        self.embeddings = wi + wj
        norm = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.unit_embeddings = self.embeddings / (norm + 1e-10)

        print(f"[OK] Loaded {lang}")
        print(f"     Vocab size: {len(self.vocab)}")
        print(f"     Embedding dim: {self.embeddings.shape[1]}")

    # ---------------------------------------------------------
    def get_word_vec(self, word: str):
        idx = self.vocab.get(word)
        if idx is None:
            return None
        return self.unit_embeddings[idx]

    # ---------------------------------------------------------
    def nearest_words(self, word: str, k=10):
        vec = self.get_word_vec(word)
        if vec is None:
            print(f"[OOV] {word}")
            return

        sims = self.unit_embeddings @ vec
        print(f"\nNearest words for '{word}' ({self.lang})")

        for i in np.argsort(-sims):
            tok = self.id_to_word[i]
            if tok != word and tok != "<UNK>":
                print(f"{tok:20s} {sims[i]:.4f}")
            if i >= k:
                break

    # ---------------------------------------------------------
    def analogy(self, a: str, b: str, c: str, k=10):
        va = self.get_word_vec(a)
        vb = self.get_word_vec(b)
        vc = self.get_word_vec(c)

        if va is None or vb is None or vc is None:
            print("[WARN] OOV in analogy")
            return

        target = va - vb + vc
        target /= np.linalg.norm(target) + 1e-10

        sims = self.unit_embeddings @ target

        print(f"\nAnalogy ({self.lang}): {a} - {b} + {c}")
        for i in np.argsort(-sims):
            tok = self.id_to_word[i]
            if tok not in {a, b, c, "<UNK>"}:
                print(f"{tok:20s} {sims[i]:.4f}")
            if i >= k:
                break


if __name__ == "__main__":
    explorer = GloVeExplorer("Bhili")

    explorer.nearest_words("पूर्ति", k=10)
    explorer.analogy("पूर्ति", "पूरा", "करवा", k=10)
