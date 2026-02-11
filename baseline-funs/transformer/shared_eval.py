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

# ================= UTILS =================

def generate_square_subsequent_mask(sz, device):
    return torch.triu(
        torch.full((sz, sz), float("-inf"), device=device),
        diagonal=1
    )

# ================= BEAM SEARCH =================

@torch.no_grad()
def beam_search(model, src, lang, device):
    """
    src: (1, S)
    returns: List[int]
    """

    model.eval()

    # ----- Encoder -----
    src = src.transpose(0, 1)  # (S, 1)
    memory = model.encoder(
        model.pe(model.emb(src))
    )

    # Beam: (tokens, score)
    beams = [([EvalConfig.BOS_ID], 0.0)]

    for _ in range(EvalConfig.MAX_LEN):
        candidates = []

        for tokens, score in beams:

            if tokens[-1] == EvalConfig.EOS_ID:
                candidates.append((tokens, score))
                continue

            tgt = torch.tensor(tokens, device=device).unsqueeze(1)  # (T, 1)
            tgt_emb = model.pe(model.emb(tgt))

            tgt_mask = generate_square_subsequent_mask(
                tgt.size(0),
                device
            )

            out = model.decoders[lang](
                tgt_emb,
                memory,
                tgt_mask=tgt_mask
            )

            logits = model.heads[lang](out[-1])  # (1, vocab)
            log_probs = F.log_softmax(logits, dim=-1)

            topk = torch.topk(log_probs, EvalConfig.BEAM_WIDTH)

            for i in range(EvalConfig.BEAM_WIDTH):
                candidates.append((
                    tokens + [topk.indices[0, i].item()],
                    score + topk.values[0, i].item()
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
            ref_txt = str(row.iloc[1])   # Gold target

            src_ids = sp.encode(
                src_txt,
                add_bos=True,
                add_eos=True
            )

            src_tensor = torch.tensor(
                [src_ids],
                device=device
            )

            pred_ids = beam_search(
                model,
                src_tensor,
                lang,
                device
            )

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
