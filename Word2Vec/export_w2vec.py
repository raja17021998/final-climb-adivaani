import torch
from pathlib import Path

BASE = Path("/home/shashwat1/final-climb-shashwat-do-not-delete/Word2Vec")

def export_all():
    for lang_dir in BASE.iterdir():
        if not lang_dir.is_dir() or lang_dir.name=="sentencepiece": continue
        pt = lang_dir/f"{lang_dir.name.lower()}_weights.pt"
        if not pt.exists(): continue
        data=torch.load(pt,weights_only=False)
        vecs=data["weights"]["target_embeddings.weight"].cpu().numpy()
        id2w=data["id_to_word"]
        with open(lang_dir/f"{lang_dir.name.lower()}.vec","w",encoding="utf-8") as f:
            f.write(f"{len(id2w)} {vecs.shape[1]}\n")
            for i,w in enumerate(id2w):
                f.write(w+" "+" ".join(f"{x:.6f}" for x in vecs[i])+"\n")
        print(f"[OK] {lang_dir.name}")

if __name__=="__main__":
    export_all()
