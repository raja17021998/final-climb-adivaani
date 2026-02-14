from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "facebook/nllb-200-distilled-600M"
# 1. Initialize tokenizer with source language
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "Education is the most powerful weapon which you can use to change the world."
tgt_lang = "sat_Olck" 

# 2. Tokenize (src_lang is now embedded in the input_ids)
inputs = tokenizer(text, return_tensors="pt")

# 3. Generate with the correct target token
translated_tokens = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
    max_length=128
)

result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
print(f"\nSource (English): {text}")
print(f"Target (Santali): {result}")
