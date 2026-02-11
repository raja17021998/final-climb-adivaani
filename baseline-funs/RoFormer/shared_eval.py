import os
import torch
import torch.nn.functional as F
import pandas as pd
import sentencepiece as spm
import evaluate
from tqdm import tqdm

from shared_train import SharedTransformer, Config

# ================= EVAL CONFIG =================

class EvalConfig(Config):
    BEAM_WIDTH = 5
    MAX_LEN = 60
    BOS_ID = 1
    EOS_ID = 2

# ================= BEAM SEARCH =================

@torch.no_grad()
def beam_search(model, src, lang, device):
    """
    src: (1, S)
    returns: list[int] (token ids)
    """

    # ----- Encoder -----
    enc = model.emb(src)
    for blk in model.encoder:
        enc = blk(enc)

    # Each beam: (tokens, score, caches)
    beams = [([EvalConfig.BOS_ID], 0.0, None)]

    for _ in range(EvalConfig.MAX_LEN):
        candidates = []

        for tokens, score, caches in beams:

            # Stop expanding finished beams
            if tokens[-1] == EvalConfig.EOS_ID:
                candidates.append((tokens, score, caches))
                continue

            curr = torch.tensor([[tokens[-1]]], device=device)
            x = model.emb(curr)

            new_caches = []
            for i, blk in enumerate(model.decoders[lang]):
                layer_cache = caches[i] if caches else None
                x, nc = blk(x, enc, cache=layer_cache)
                new_caches.append(nc)

            logits = F.log_softmax(
                model.heads[lang](x[:, -1, :]),
                dim=-1
            )

            topk = torch.topk(logits, EvalConfig.BEAM_WIDTH)

            for i in range(EvalConfig.BEAM_WIDTH):
                candidates.append((
                    tokens + [topk.indices[0, i].item()],
                    score + topk.values[0, i].item(),
                    new_caches
                ))

        beams = sorted(candidates, key=lambda x: x[1], reverse=True)
        beams = beams[:EvalConfig.BEAM_WIDTH]

        if all(b[0][-1] == EvalConfig.EOS_ID for b in beams):
            break

    return beams[0][0]

# ================= MAIN EVAL =================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sp = spm.SentencePieceProcessor(model_file=EvalConfig.SPM_PATH)

    model = SharedTransformer().to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(EvalConfig.SAVE_DIR, "shared.pt"),
            map_location=device
        )
    )
    model.eval()

    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    for lang in EvalConfig.LANGS:
        print(f"\n🔍 Evaluating: {lang}")

        csv_path = os.path.join(
            EvalConfig.DATASET_DIR,
            EvalConfig.CSV_FILES[lang]
        )

        df = pd.read_csv(csv_path).head(100)
        results = []

        for _, row in tqdm(df.iterrows(), total=len(df)):

            src_txt = str(row.iloc[0])   # Hindi
            ref_txt = str(row.iloc[1])   # Gold Bhili

            src_ids = sp.encode(
                src_txt,
                add_bos=True,
                add_eos=True
            )

            src_tensor = torch.tensor(
                [src_ids],
                device=device
            )

            pred_ids = beam_search(model, src_tensor, lang, device)

            # Remove special tokens
            pred_ids = [
                t for t in pred_ids
                if t not in (EvalConfig.BOS_ID, EvalConfig.EOS_ID)
            ]

            pred_txt = sp.decode(pred_ids)

            # ---- Sentence BLEU ----
            bleu = bleu_metric.compute(
                predictions=[pred_txt],
                references=[[ref_txt]]
            )["score"]

            # ---- Sentence CHRF++ ----
            chrf = chrf_metric.compute(
                predictions=[pred_txt],
                references=[[ref_txt]]
            )["score"]

            results.append([
                src_txt,
                ref_txt,
                pred_txt,
                bleu,
                chrf
            ])

        out_df = pd.DataFrame(
            results,
            columns=[
                "Actual Hindi",
                "Actual bhili",
                "Predicted bhili",
                "Bleu",
                "Chrf++"
            ]
        )

        out_path = f"eval_{lang}.csv"
        out_df.to_csv(out_path, index=False)

        print(f"✅ Saved → {out_path}")

# ================= ENTRY =================

if __name__ == "__main__":
    main()
