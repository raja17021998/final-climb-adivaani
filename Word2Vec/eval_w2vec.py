import torch, numpy as np, sentencepiece as spm
from pathlib import Path

BASE=Path("/home/shashwat1/final-climb-shashwat-do-not-delete/Word2Vec")

def eval_lang(lang,word):
    data=torch.load(BASE/lang/f"{lang.lower()}_weights.pt",weights_only=False)
    emb=data["weights"]["target_embeddings.weight"].cpu().numpy()
    unit=emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-10)
    w2id=data["vocab"]; id2w=data["id_to_word"]
    sp=spm.SentencePieceProcessor()
    sp.load(str(BASE/"sentencepiece"/f"{lang.lower()}_sp.model"))
    pcs=sp.encode(word,out_type=str)
    vecs=[emb[w2id[p]] for p in pcs if p in w2id]
    v=np.mean(vecs,0); v/=np.linalg.norm(v)+1e-10
    sims=unit@v
    for i in np.argsort(-sims)[:10]:
        print(id2w[i],sims[i])

if __name__=="__main__":
    eval_lang("Bhili","पूर्ति")
