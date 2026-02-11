import pandas as pd
import sentencepiece as spm
import os

def prepare_tokenizer():
    # Jahan output save karna hai
    shared_dir = "/home/user/Desktop/Shashwat/final-climb/baseline-funs"
    os.makedirs(shared_dir, exist_ok=True)
    
    # Jahan datasets rakhe hain
    dataset_dir = "/home/user/Desktop/Shashwat/final-climb/datasets"
    
    csv_files = [
        "Hin_Bhi_Mar_Guj.csv", 
        "Hin_Gon.csv", 
        "Hin_Kui.csv", 
        "Hin_Mun.csv"
    ]
    
    all_text = []
    found_any_file = False
    
    for f_name in csv_files:
        f_path = os.path.join(dataset_dir, f_name)
        if os.path.exists(f_path):
            print(f"Loading {f_path}...")
            df = pd.read_csv(f_path)
            # Hindi aur Tribal columns ka text extract kar rahe hain
            for col in df.columns:
                all_text.extend(df[col].astype(str).tolist())
            found_any_file = True
        else:
            print(f"Warning: File not found -> {f_path}")
    
    if not found_any_file:
        print(f"Error: {dataset_dir} mein ek bhi CSV nahi mili! Path check karo.")
        return

    corpus_path = os.path.join(shared_dir, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))
    
    print(f"Corpus created with {len(all_text)} lines at {corpus_path}")

    # Training - Vocab size 8000 as discussed
    spm.SentencePieceTrainer.train(
        input=corpus_path, 
        model_prefix=os.path.join(shared_dir, "joint_spm"),
        vocab_size=32000, 
        model_type='unigram', 
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
        character_coverage=0.9995,
        num_threads=16
    )
    print(f"Success: joint_spm.model saved in {shared_dir}")

if __name__ == "__main__":
    prepare_tokenizer()