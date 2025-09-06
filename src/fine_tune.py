#!/usr/bin/env python
"""
LoRA fine-tuning for LLaMA-3.1-8B (4-bit) using PEFT + Hugging Face Trainer.
- Uses YAML prompt template
- Reads CSV (business_description, domain_names)
- Logs training/eval loss
- Supports early stopping
- Rich progress bar
"""
import ast
import json
import os

import pandas as pd
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from rich.progress import track
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

# -----------------------------
# 0. Configuration (edit these)
# -----------------------------
CSV_PATH = "domains.csv"
PROMPT_YAML_PATH = "prompts/prompt-1.yaml"
OUTPUT_DIR = "./lora-mistral-domain"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
EVAL_RATIO = 0.1
SEED = 42

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.manual_seed(SEED)

# -----------------------------
# 1. load prompt
# -----------------------------
with open(PROMPT_YAML_PATH, "r", encoding="utf-8") as f:
    prompt_yaml = yaml.safe_load(f)
PROMPT_TEMPLATE = prompt_yaml["domain_prompt"]

# -----------------------------
# 2. load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# -----------------------------
# 3. load CSV -> Dataset
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
        completion = json.dumps(domains)
        text = prompt + "\n" + completion + tokenizer.eos_token
        rows.append({"prompt": prompt, "completion": completion, "text": text})
    return Dataset.from_list(rows)


dataset = csv_to_dataset(CSV_PATH, PROMPT_TEMPLATE)
dataset = dataset.train_test_split(test_size=EVAL_RATIO, seed=SEED)
train_ds = dataset["train"]
eval_ds = dataset["test"]


# -----------------------------
# 4. tokenization
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


train_tokenized = train_ds.map(
    tokenize_fn, batched=True, remove_columns=train_ds.column_names
)
eval_tokenized = eval_ds.map(
    tokenize_fn, batched=True, remove_columns=eval_ds.column_names
)

# -----------------------------
# 5. load 4-bit model
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

# -----------------------------
# 6. apply LoRA
# -----------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------------
# 7. data collator & Trainer
# -----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    save_total_limit=3,
    gradient_checkpointing=True,
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
)

# -----------------------------
# 8. train
# -----------------------------
trainer.train()
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))

print(
    f"✅ Training finished. LoRA adapter saved to "
    f"{os.path.join(OUTPUT_DIR, 'lora_adapter')}"
)
