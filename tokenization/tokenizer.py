import pandas as pd
import sentencepiece as spm
import os

def prepare_tokenizer():
    base_dir = "/home/jovyan/final-climb-shashwat-do-not-delete/"
    
    # Shared tokenizer artifacts
    shared_dir = f"{base_dir}/tokenization"
    os.makedirs(shared_dir, exist_ok=True)

    # Language-specific corpora directory
    corpora_dir = f"{base_dir}/tokenization/corpora"
    os.makedirs(corpora_dir, exist_ok=True)
    
    dataset_dir = f"{base_dir}/datasets"
    
    csv_files = [
        "Bhi_Hin_Mar_Guj_Eng.csv",
        "Garo_Hin_Eng.csv",
        "Gon_Hin_Eng.csv",
        "Kui_Hin_Eng.csv",
        "Mun_Hin_Eng.csv",
        "San_Hin_Eng.csv",
    ]
    
    # Mapping CSV → language key + column name hint
    lang_map = {
        "Bhi_Hin_Mar_Guj_Eng.csv": ("bhili", "Bhi"),
        "San_Hin_Eng.csv": ("santali", "San"),
        "Mun_Hin_Eng.csv": ("mundari", "Mun"),
        "Gon_Hin_Eng.csv": ("gondi", "Gon"),
        "Kui_Hin_Eng.csv": ("kui", "Kui"),
        "Garo_Hin_Eng.csv": ("garo", "Garo"),  # optional if needed
    }

    all_text = []
    lang_corpora = {k: [] for k, _ in lang_map.values()}
    found_any_file = False
    
    for f_name in csv_files:
        f_path = os.path.join(dataset_dir, f_name)
        if not os.path.exists(f_path):
            print(f"Warning: File not found -> {f_path}")
            continue

        print(f"Loading {f_path}...")
        df = pd.read_csv(f_path)

        # Add ALL text for joint tokenizer
        for col in df.columns:
            all_text.extend(df[col].astype(str).tolist())

        # Extract language-specific column if present
        lang_key, col_hint = lang_map[f_name]
        for col in df.columns:
            if col_hint.lower() in col.lower():
                lang_corpora[lang_key].extend(df[col].astype(str).tolist())

        found_any_file = True

    if not found_any_file:
        print(f"Error: No CSV files found in {dataset_dir}")
        return

    # ------------------------------------------------------------
    # Save merged multilingual corpus (for tokenizer training)
    # ------------------------------------------------------------
    corpus_path = os.path.join(shared_dir, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))
    
    print(f"Joint corpus created with {len(all_text)} lines at {corpus_path}")

    # ------------------------------------------------------------
    # Save per-language corpora (for separate BERT training)
    # ------------------------------------------------------------
    for lang, texts in lang_corpora.items():
        out_path = os.path.join(corpora_dir, f"{lang}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join([t for t in texts if str(t).strip() != ""]))
        print(f"{lang} corpus saved → {out_path} ({len(texts)} lines)")

    # ------------------------------------------------------------
    # Train shared SentencePiece tokenizer
    # ------------------------------------------------------------
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=os.path.join(shared_dir, "joint_spm"),
        vocab_size=32000,
        model_type="unigram",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols=["<mask>"],
        character_coverage=0.9995,
        num_threads=16
    )

    print(f"Success: joint_spm.model saved in {shared_dir}")

if __name__ == "__main__":
    prepare_tokenizer()
