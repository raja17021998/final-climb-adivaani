import torch
import numpy as np
from pathlib import Path

# ============================================================
# PATH
# ============================================================

BASE = Path("/home/jovyan/final-climb-shashwat-do-not-delete/Word2Vec")


def export_all():

    for lang_dir in BASE.iterdir():

        if not lang_dir.is_dir():
            continue

        pt = lang_dir / f"{lang_dir.name.lower()}_weights.pt"

        if not pt.exists():
            continue

        print(f"Exporting {lang_dir.name}...")

        data = torch.load(pt, map_location="cpu")

        # 🔥 Correct key for your SGNS model
        vecs = data["weights"]["t.weight"].cpu().numpy()
        id2w = data["id_to_word"]

        out_path = lang_dir / f"{lang_dir.name.lower()}.vec"

        with open(out_path, "w", encoding="utf-8") as f:

            # Header line (word2vec format)
            f.write(f"{len(id2w)} {vecs.shape[1]}\n")

            for i, word in enumerate(id2w):
                vector_str = " ".join(f"{x:.6f}" for x in vecs[i])
                f.write(f"{word} {vector_str}\n")

        print(f"[OK] Saved to {out_path}")


if __name__ == "__main__":
    export_all()
