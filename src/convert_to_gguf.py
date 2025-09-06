#!/usr/bin/env python
"""
Merge LoRA adapter into the base model and prepare for GGUF export for llama.cpp
"""

import os

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# -----------------------------
# Configuration (edit these)
# -----------------------------
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8b"  # path or HF repo of base model
LORA_ADAPTER_PATH = "./lora-llama3-domain/lora_adapter"  # LoRA adapter output
MERGED_OUTPUT_DIR = "./llama3_8b_merged"  # merged HF model output

# Create output dir if not exists
os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1. Load LoRA + merge
# -----------------------------
print("Loading LoRA adapter...")
model = AutoPeftModelForCausalLM.from_pretrained(
    LORA_ADAPTER_PATH, device_map="auto"
)

print("Merging LoRA adapter into base weights...")
model = model.merge_and_unload()

# Save merged HF model
print(f"Saving merged model to {MERGED_OUTPUT_DIR} ...")
model.save_pretrained(MERGED_OUTPUT_DIR)

# -----------------------------
# 2. Save tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

print("✅ Merged model ready.")
print(
    "Next step: use llama.cpp convert.py to GGUF, then quantize if desired (Q4_K_M)."
)
