#!/usr/bin/env python
"""
LoRA fine-tuning for Mistral-7B-Instruct-v0.3 (4-bit) using PEFT + Hugging Face Trainer.
- Uses YAML prompt template
- Reads CSV (business_description, domain_names)
- Logs training/eval loss
- Supports early stopping
- Rich progress bar
- Automatically detects attention projection layers for LoRA
"""
import ast
import json
import os

import pandas as pd
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from rich.progress import Progress, track
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

# -----------------------------
# 0. Configuration
# -----------------------------
CSV_PATH = "data/raw/train_dataset.csv"
PROMPT_YAML_PATH = "prompts/prompt-1.yaml"
OUTPUT_DIR = "./lora-mistral-domain"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1
NUM_EPOCHS = 4
LEARNING_RATE = 1.7e-4
EVAL_RATIO = 0.1
SEED = 42

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.manual_seed(SEED)

# -----------------------------
# 1. Load prompt
# -----------------------------
with open(PROMPT_YAML_PATH, "r", encoding="utf-8") as f:
    prompt_yaml = yaml.safe_load(f)
PROMPT_TEMPLATE = prompt_yaml["domain_prompt"]

# -----------------------------
# 2. Load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# -----------------------------
# 3. Load CSV -> Dataset
# -----------------------------
def csv_to_dataset(csv_path, prompt_template):
    df = pd.read_csv(csv_path)
    rows = []
    for idx in track(range(len(df)), description="Processing CSV"):
        row = df.iloc[idx]
        description = str(row["business_description"])
        try:
            domains = ast.literal_eval(row["domain_names"])
        except (ValueError, SyntaxError):
            domains = row["domain_names"]
        prompt = prompt_template.format(description=description)
        completion = json.dumps(domains)  # During v1.0 this was str()
        text = prompt + "\n" + completion + tokenizer.eos_token
        rows.append({"prompt": prompt, "completion": completion, "text": text})
    return Dataset.from_list(rows)


dataset = csv_to_dataset(CSV_PATH, PROMPT_TEMPLATE)
dataset = dataset.train_test_split(test_size=EVAL_RATIO, seed=SEED)
train_ds = dataset["train"]
eval_ds = dataset["test"]


# -----------------------------
# 4. Tokenization
# -----------------------------
def tokenize_fn(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


with Progress() as progress:
    t1 = progress.add_task(
        "[green]Tokenizing train dataset...", total=len(train_ds)
    )
    train_tokenized = train_ds.map(
        lambda x: tokenize_fn(x),
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train dataset",
    )
    progress.update(t1, advance=len(train_ds))

    t2 = progress.add_task(
        "[blue]Tokenizing eval dataset...", total=len(eval_ds)
    )
    eval_tokenized = eval_ds.map(
        lambda x: tokenize_fn(x),
        batched=True,
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval dataset",
    )
    progress.update(t2, advance=len(eval_ds))

# -----------------------------
# 5. Load 4-bit model
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.config.use_cache = False

# -----------------------------
# 6. Detect LoRA target modules
# -----------------------------
target_modules = [
    name
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear)
    and ("q_proj" in name or "v_proj" in name)
]
print(f"Detected {len(target_modules)} LoRA target modules:")
for m in target_modules:
    print("  -", m)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
# Important for 4-bit + gradient checkpointing:
# Enables gradients to flow from the input embeddings to the LoRA layers.
# Without this, PyTorch cannot compute gradients for LoRA adapters,
# and you’ll get the error: "element 0 of tensors does not require grad".
model.enable_input_require_grads()
model.print_trainable_parameters()

# -----------------------------
# 7. Data collator & Trainer
# -----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    # The two arguments from below weren't used in v1.0
    dataloader_num_workers=64,  # number of CPU processes for loading batches
    dataloader_pin_memory=True,  # pins memory for faster CPU→GPU transfer
    fp16=True,
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    save_total_limit=3,
    # changed next attrib in v2.1 from True to False
    # (Roughly 35GB instead of 4GB VRAM but 6.5s instead of 9s p.i.)
    gradient_checkpointing=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    optimizers=(None, None),  # use default AdamW for trainable params only,
)

# -----------------------------
# 8. Train
# -----------------------------
trainer.train()

# -----------------------------
# 9. Save LoRA adapter
# -----------------------------
adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

print(f"✅ Training finished. LoRA adapter saved to {adapter_dir}")
