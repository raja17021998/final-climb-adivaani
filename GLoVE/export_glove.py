import torch
from config import *

def export_all():
    for lang in LANGUAGES:
        ckpt = torch.load(SAVE_ROOT / lang / f"{lang}_glove.pt", weights_only=False)
        wi = ckpt["weights"]["wi.weight"].cpu().numpy()
        wj = ckpt["weights"]["wj.weight"].cpu().numpy()
        vecs = wi + wj

        out = SAVE_ROOT / lang / f"{lang}.vec"
        with open(out,"w",encoding="utf-8") as f:
            f.write(f"{vecs.shape[0]} {vecs.shape[1]}\n")
            for i in range(vecs.shape[0]):
                f.write(f"{i} " + " ".join(f"{x:.6f}" for x in vecs[i]) + "\n")

        print("Exported", lang)

if __name__ == "__main__":
    export_all()
