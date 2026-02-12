import torch
import numpy as np
from pathlib import Path
import sentencepiece as spm


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/home/jovyan/final-climb-shashwat-do-not-delete/Word2Vec")
SP_MODEL_PATH = Path(
    "/home/jovyan/final-climb-shashwat-do-not-delete/tokenization/joint_spm.model"
)


class MultiLangExplorer:
    def __init__(self, lang: str):
        """
        Explorer for SentencePiece-based Word2Vec embeddings.
        Works for ALL tribal languages.
        """

        self.lang = lang
        self.lang_dir = BASE_DIR / lang
        model_path = self.lang_dir / f"{lang.lower()}_weights.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")

        if not SP_MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing joint SentencePiece model: {SP_MODEL_PATH}")

        # ====================================================
        # Load Word2Vec weights
        # ====================================================
        data = torch.load(model_path, map_location="cpu")

        self.vocab = data["vocab"]
        self.id_to_word = data["id_to_word"]

        # 🔥 Correct key for your SGNS model
        self.embeddings = data["weights"]["t.weight"].cpu().numpy()

        # Normalize embeddings for cosine similarity
        norm = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.unit_embeddings = self.embeddings / (norm + 1e-10)

        # ====================================================
        # Load Joint SentencePiece
        # ====================================================
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(SP_MODEL_PATH))

        print(f"[OK] Loaded {lang}")
        print(f"     Vocab size: {len(self.vocab)}")
        print(f"     Embedding dim: {self.embeddings.shape[1]}")

    # ---------------------------------------------------------
    # Subword embedding
    # ---------------------------------------------------------
    def get_subword_vec(self, token: str):
        idx = self.vocab.get(token)
        if idx is None:
            return None
        return self.embeddings[idx]

    # ---------------------------------------------------------
    # Word embedding (SentencePiece aggregation)
    # ---------------------------------------------------------
    def get_word_embedding(self, word: str, normalize: bool = True):
        pieces = self.sp.encode(word, out_type=str)
        if not pieces:
            return None

        vecs = []
        for p in pieces:
            v = self.get_subword_vec(p)
            if v is not None:
                vecs.append(v)

        if not vecs:
            return None

        word_vec = np.mean(vecs, axis=0)

        if normalize:
            word_vec /= np.linalg.norm(word_vec) + 1e-10

        return word_vec

    # ---------------------------------------------------------
    # Nearest neighbors (subword space)
    # ---------------------------------------------------------
    def nearest_tokens(self, vec, k=10, exclude=None):
        if exclude is None:
            exclude = set()

        vec = vec / (np.linalg.norm(vec) + 1e-10)
        sims = np.dot(self.unit_embeddings, vec)

        results = []
        for idx in np.argsort(-sims):
            token = self.id_to_word[idx]
            if token not in exclude and token != "<UNK>":
                results.append((token, sims[idx]))
            if len(results) >= k:
                break
        return results

    # ---------------------------------------------------------
    # Word-level nearest neighbors
    # ---------------------------------------------------------
    def nearest_words(self, word: str, k=10):
        vec = self.get_word_embedding(word)
        if vec is None:
            print(f"[WARN] OOV word: {word}")
            return

        print(f"\nNearest tokens for word: {word}")
        for tok, score in self.nearest_tokens(vec, k=k):
            print(f"{tok:20s} {score:.4f}")

    # ---------------------------------------------------------
    # Analogy: a - b + c
    # ---------------------------------------------------------
    def analogy(self, a: str, b: str, c: str, k=10):
        va = self.get_word_embedding(a)
        vb = self.get_word_embedding(b)
        vc = self.get_word_embedding(c)

        if va is None or vb is None or vc is None:
            missing = [w for w, v in zip([a, b, c], [va, vb, vc]) if v is None]
            print(f"[WARN] Missing embeddings for: {missing}")
            return

        target_vec = va - vb + vc

        print(f"\nAnalogy ({self.lang}): {a} - {b} + {c}")
        for tok, score in self.nearest_tokens(
            target_vec,
            k=k,
            exclude=set(self.sp.encode(a, out_type=str)),
        ):
            print(f"{tok:20s} {score:.4f}")


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    # Try any tribal language
    for lang in ["Bhili", "Santali", "Mundari", "Gondi", "Kui", "Garo"]:

        print("\n==============================")
        print(f"Testing {lang}")
        print("==============================")

        explorer = MultiLangExplorer(lang)

        explorer.nearest_words("पूरा", k=5)
