# model_utils.py
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
from config import *

def is_dist():
    return dist.is_available() and dist.is_initialized()

def broadcast_tokenizer(tokenizer, src=0):
    if not is_dist():
        return tokenizer
    obj_list = [tokenizer]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]

def _init_embedding_from_refs(tokenizer, model, new_token, ref_tokens):
    emb = model.get_input_embeddings().weight.data
    new_id = tokenizer.convert_tokens_to_ids(new_token)
    ref_ids = [tokenizer.convert_tokens_to_ids(t) for t in ref_tokens if t in tokenizer.get_vocab()]
    if ref_ids:
        emb[new_id] = emb[ref_ids].mean(dim=0)

def resolve_language_tokens(tokenizer, model, rank):
    # Case A & B: Existing script / new language token
    if rank == 0:
        added_tokens = []

        for lang, code in LANGUAGE_CODE_MAP.items():
            if code not in tokenizer.get_vocab():
                tokenizer.add_special_tokens({"additional_special_tokens": [code]})
                added_tokens.append(code)

        # Case C: Completely new scripts from config
        for new_code, cfg in NEW_SCRIPT_LANGS.items():
            if new_code not in tokenizer.get_vocab():
                tokenizer.add_special_tokens({"additional_special_tokens": [new_code]})
                added_tokens.append(new_code)

        if added_tokens:
            model.resize_token_embeddings(len(tokenizer))

            # Initialize embeddings
            for token in added_tokens:
                if token in LANGUAGE_CODE_MAP.values():
                    script = token.split("_")[-1]
                    ref_tokens = SCRIPT_MAP.get(script, ["hin_Deva", "eng_Latn"])
                    _init_embedding_from_refs(tokenizer, model, token, ref_tokens)

                if token in NEW_SCRIPT_LANGS:
                    ref_tokens = NEW_SCRIPT_LANGS[token].get("init_from", ["hin_Deva", "eng_Latn"])
                    _init_embedding_from_refs(tokenizer, model, token, ref_tokens)

    tokenizer = broadcast_tokenizer(tokenizer, 0)
    return tokenizer, model

def build_model_and_tokenizer(tribal, rank):
    model_name = MODEL_CHOICE[tribal]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenizer, model = resolve_language_tokens(tokenizer, model, rank)

    if USE_LORA[tribal]:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        model = get_peft_model(model, lora_config)

    return tokenizer, model
