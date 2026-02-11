import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from text_protection import protect_text, restore_text

# ======================================================
# PATHS
# ======================================================
BASE_DIR = "/home/kshhorizon/data/final-climb-shashwat-do-not-delete/foundation-model"

# ======================================================
# MODEL MAP
# ======================================================
MODEL_MAP = {
    "nllb-200-600m": "facebook/nllb-200-600M",
    "nllb-200-1.3B": "facebook/nllb-200-1.3B",
    "nllb-200-3.3B": "facebook/nllb-200-3.3B",
}

TRIBAL_PREFIXES = ["bhi", "mun", "gon", "san", "gar", "kui"]

# ======================================================
# ARGS
# ======================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_variant",
    choices=list(MODEL_MAP.keys()),
    required=True,
)
parser.add_argument(
    "--use_lora",
    action="store_true",
    help="Load LoRA adapter (ONLY valid for nllb-200-3.3B)",
)

args = parser.parse_args()

# ======================================================
# VALIDATION
# ======================================================
if args.use_lora and args.model_variant != "nllb-200-3.3B":
    raise ValueError("--use_lora is only supported for nllb-200-3.3B")

# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if device.type == "cuda":
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")

# ======================================================
# RESOLVE MODEL DIR
# ======================================================
variant_name = args.model_variant.replace(".", "_")
suffix = "lora" if args.use_lora else "full"

MODEL_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    f"{variant_name}_{suffix}",
)

print(f"📦 Loading model from: {MODEL_DIR}")

# ======================================================
# LOAD TOKENIZER
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

# ======================================================
# LOAD MODEL
# ======================================================
if args.use_lora:
    from peft import PeftModel

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_MAP[args.model_variant],
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        use_safetensors=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        MODEL_DIR,
        local_files_only=True,
    )
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        use_safetensors=True,
        local_files_only=True,
    )

model.to(device).eval()

# ======================================================
# HELPERS
# ======================================================
def is_tribal(lang_token: str) -> bool:
    return any(lang_token.startswith(p) for p in TRIBAL_PREFIXES)

def infer_direction(src_lang: str, tgt_lang: str) -> str:
    if is_tribal(tgt_lang) and not is_tribal(src_lang):
        return "hub_to_tribal"
    if is_tribal(src_lang) and not is_tribal(tgt_lang):
        return "tribal_to_hub"
    return "other"

def beam_size_for_direction(direction: str) -> int:
    # Consistent with evaluation policy
    return 5 if direction == "hub_to_tribal" else 3

# ======================================================
# INTERACTIVE INFERENCE LOOP
# ======================================================
print("\n💬 Interactive NLLB inference (Ctrl+C to exit)\n")

while True:
    try:
        src = input("Source sentence: ").strip()
        sl = input("Source lang token (e.g. hin_Deva): ").strip()
        tl = input("Target lang token (e.g. san_Olck): ").strip()

        protected, mapping = protect_text(src)

        tokenizer.src_lang = sl
        tokenizer.tgt_lang = tl

        direction = infer_direction(sl, tl)
        beam_size = beam_size_for_direction(direction)

        tgt_id = tokenizer.convert_tokens_to_ids(tl)

        inputs = tokenizer(
            protected,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=128,
                num_beams=beam_size,
                forced_bos_token_id=tgt_id,
                early_stopping=True,
            )

        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        restored = restore_text(pred, mapping)

        print(f"\n🧭 Direction: {direction}")
        print(f"🔍 Beam size: {beam_size}")
        print("🟢 Output:", restored, "\n")

    except KeyboardInterrupt:
        print("\n👋 Exiting inference.")
        break
