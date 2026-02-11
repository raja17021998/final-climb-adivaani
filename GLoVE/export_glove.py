# ============================================================
# Export GLoVe vectors (.vec)
# ============================================================

import torch
from pathlib import Path

BASE = Path("/home/shashwat1/final-climb-shashwat-do-not-delete/GLoVe")


def export_all():
    for lang_dir in BASE.iterdir():
        if not lang_dir.is_dir():
            continue

        pt = lang_dir / f"{lang_dir.name.lower()}_glove.pt"
        if not pt.exists():
            continue

        data = torch.load(pt, weights_only=False)

        wi = data["weights"]["wi.weight"].cpu().numpy()
        wj = data["weights"]["wj.weight"].cpu().numpy()
        vecs = wi + wj

        id2w = data["id_to_word"]

        out_path = lang_dir / f"{lang_dir.name.lower()}.vec"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{len(id2w)} {vecs.shape[1]}\n")
            for i, w in enumerate(id2w):
                f.write(
                    w + " " + " ".join(f"{x:.6f}" for x in vecs[i]) + "\n"
                )

        print(f"[OK] Exported {lang_dir.name}")


if __name__ == "__main__":
    export_all()
