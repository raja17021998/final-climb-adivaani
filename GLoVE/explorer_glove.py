import torch
import numpy as np
import sentencepiece as spm
from config import *

class GloVeExplorer:
    def __init__(self, lang):
        ckpt = torch.load(SAVE_ROOT / lang / f"{lang}_glove.pt", weights_only=False)
        self.emb = ckpt["weights"]["wi.weight"].cpu().numpy() + \
                   ckpt["weights"]["wj.weight"].cpu().numpy()
        self.emb /= (np.linalg.norm(self.emb, axis=1, keepdims=True)+1e-10)

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(TOKENIZER_PATH))

    def nearest(self, text, k=10):
        ids = self.sp.encode(text)
        for tid in ids:
            vec = self.emb[tid]
            sims = self.emb @ vec
            top = np.argsort(-sims)[:k]
            print(f"\nToken: {self.sp.id_to_piece(tid)}")
            for t in top:
                print(self.sp.id_to_piece(t), sims[t])

if __name__ == "__main__":
    GloVeExplorer("bhili").nearest("कम")
