# import os
# from huggingface_hub import snapshot_download

# # Directory where all models will be stored
# BASE_DIR = "/home/user/Desktop/Shashwat/final-climb/BERT-based-models"

# # List of HuggingFace model IDs to download
# model_list = {
#     "hindi-bert":                    "l3cube-pune/hindi-bert-scratch",
#     "marathi-bert":                          "l3cube-pune/marathi-bert",
#     "gujarati-bert":                        "l3cube-pune/gujarati-bert",
#     "odia-bert":                             "l3cube-pune/odia-bert",
#     "bengali-bert":                          "l3cube-pune/bengali-bert",
#     "telugu-bert":                           "l3cube-pune/telugu-bert",
#     "m-bert":              "google-bert/bert-base-multilingual-cased", # Added m-BERT
# }

# def download_models(base_dir, models):
#     os.makedirs(base_dir, exist_ok=True)

#     for local_name, hf_id in models.items():
#         print("="*80)
#         print(f"⬇️ Downloading {hf_id} ...")
        
#         # each model goes in its own directory
#         target_dir = os.path.join(base_dir, local_name)

#         # Download full repo snapshot
#         try:
#             snapshot_download(
#                 repo_id=hf_id,
#                 repo_type="model",
#                 cache_dir=target_dir,
#                 local_dir=target_dir,
#                 local_dir_use_symlinks=False,
#             )
#             print(f"✅ Saved {local_name} → {target_dir}")
#         except Exception as e:
#             print(f"❌ Failed at {hf_id}: {e}")

#     print("="*80)
#     print("🎉 All downloads complete.")

# if __name__ == "__main__":
#     download_models(BASE_DIR, model_list)


#!/usr/bin/env python3
"""
Robust BERT downloader for Indic + mBERT models.
- Works behind institute CA setups
- Reuses local snapshots if present
- Clean logging
"""




import os
import shutil
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "/home/kshhorizon/data/final-climb-shashwat-do-not-delete/Seq2Seq-based-models"

MODELS = {
    "nllb-200-600m": "facebook/nllb-200-distilled-600M",
    "nllb-200-1.3b": "facebook/nllb-200-1.3B",
    "nllb-200-3.3b": "facebook/nllb-200-3.3B",
}

BIN_ALLOW = [
    "*.bin",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

DTYPE = torch.bfloat16   # A100-safe

# ============================================================
def download_bin(repo_id, target_dir):
    print(f"⬇️  Downloading BIN weights for {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        allow_patterns=BIN_ALLOW,
    )


def convert_to_safetensors(src_dir, dst_dir):
    print(f"🔁 Converting → safetensors")
    os.makedirs(dst_dir, exist_ok=True)

    # Load BIN model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        src_dir,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        device_map="cpu",   # safest for conversion
    )

    # Save SAFETENSORS
    model.save_pretrained(
        dst_dir,
        safe_serialization=True,
    )

    # Copy tokenizer + configs
    tokenizer = AutoTokenizer.from_pretrained(src_dir)
    tokenizer.save_pretrained(dst_dir)

    del model
    torch.cuda.empty_cache()


def cleanup_bin(dir_path):
    print(f"🧹 Removing .bin shards")
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.endswith(".bin"):
                os.remove(os.path.join(root, f))


# ============================================================
if __name__ == "__main__":

    for name, repo in MODELS.items():
        print("=" * 80)

        bin_dir = os.path.join(BASE_DIR, f"{name}-bin")
        safe_dir = os.path.join(BASE_DIR, name)

        # Skip if safetensors already exist
        if os.path.exists(os.path.join(safe_dir, "model.safetensors")):
            print(f"✅ Safetensors already exist for {name}, skipping.")
            continue

        os.makedirs(bin_dir, exist_ok=True)

        download_bin(repo, bin_dir)
        convert_to_safetensors(bin_dir, safe_dir)
        cleanup_bin(bin_dir)

        print(f"✅ DONE: {name}")

    print("=" * 80)
    print("🎉 All models downloaded & converted to safetensors.")
