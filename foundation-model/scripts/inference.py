# inference.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from text_protection import protect_text, restore_text
import config

_model = None
_tok = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# LANGUAGE INFERENCE
# ======================================================
def _infer_lang(token, table):
    for name, prefixes in table.items():
        if any(token.startswith(p) for p in prefixes):
            return name
    return None


def _infer_route(src_lang, tgt_lang):
    sh = _infer_lang(src_lang, config.HUB_LANGS)
    th = _infer_lang(tgt_lang, config.HUB_LANGS)
    st = _infer_lang(src_lang, config.TRIBAL_LANGS)
    tt = _infer_lang(tgt_lang, config.TRIBAL_LANGS)

    if sh and tt:
        return "hub_to_tribal"
    if st and th:
        return "tribal_to_hub"
    return "other"


# ======================================================
# LOAD MODEL (LAZY)
# ======================================================
def _load():
    global _model, _tok

    if _model is not None:
        return

    variant = config.DEFAULT_MODEL_VARIANT.replace(".", "_")
    suffix = "lora" if config.LORA_ENABLED else "full"

    model_dir = os.path.join(
        config.OUTPUT_DIR,
        f"{variant}_{suffix}",
        "final_model_weight"
    )

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found:\n{model_dir}")

    _tok = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    if config.LORA_ENABLED:
        from peft import PeftModel

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            config.MODEL_MAP[config.DEFAULT_MODEL_VARIANT],
            torch_dtype=torch.bfloat16 if _device.type == "cuda" else torch.float32,
        )

        _model = PeftModel.from_pretrained(
            base_model,
            model_dir,
            local_files_only=True,
        )

    else:
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16 if _device.type == "cuda" else torch.float32,
            local_files_only=True,
        )

    _model.to(_device).eval()


# ======================================================
# TRANSLATE
# ======================================================
def translate(sentence: str, src_lang: str, tgt_lang: str) -> str:
    _load()

    if not sentence or not sentence.strip():
        return ""

    protected, mapping = protect_text(sentence)

    _tok.src_lang = src_lang
    _tok.tgt_lang = tgt_lang

    route = _infer_route(src_lang, tgt_lang)
    beam = config.BEAM_BY_ROUTE.get(route, config.DEFAULT_BEAM)

    inputs = _tok(
        protected,
        return_tensors="pt",
        truncation=True,
        max_length=config.MAX_LEN,
    ).to(_device)

    forced_id = _tok.convert_tokens_to_ids(tgt_lang)

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            num_beams=beam,
            forced_bos_token_id=forced_id,
            max_length=config.MAX_LEN,
        )

    decoded = _tok.decode(output[0], skip_special_tokens=True)
    return restore_text(decoded, mapping)
