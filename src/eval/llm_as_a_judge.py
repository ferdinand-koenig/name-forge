# 📌 Imports
import json

import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

tqdm.pandas()

# 📌 Load YAML prompt
with open("prompts/judge_prompt.yaml", "r") as f:
    yaml_content = yaml.safe_load(f)

JUDGE_PROMPT_TEMPLATE = yaml_content["judge_prompt"]

# 📌 Load Qwen2.5 model (multi-GPU)
model_name = "Qwen/Qwen2.5-32B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # auto distributes across GPUs
    torch_dtype=torch.float16,
)
model.eval()


# 📌 Function to evaluate a single domain using apply_chat_template
def evaluate_domain(domain: str, description: str, max_new_tokens: int = 400):
    """
    Evaluates a domain using Qwen2.5 and the YAML rubric prompt.
    Returns a JSON/dict with the 4 criteria and total_score.
    """
    # Fill YAML prompt
    prompt_text = JUDGE_PROMPT_TEMPLATE.format(
        domain=domain, description=description
    )

    # Qwen-specific chat template
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, a helpful assistant. "
            "Follow the YAML rubric provided carefully.",
        },
        {"role": "user", "content": prompt_text},
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize and move to model device
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

    # Strip prompt tokens
    generated_ids_clean = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode
    output_text = tokenizer.batch_decode(
        generated_ids_clean, skip_special_tokens=True
    )[0]

    # Extract JSON safely
    try:
        start = output_text.find("{")
        end = output_text.rfind("}") + 1
        if start == -1 or end == -1:
            raise ValueError("JSON not found in model output.")
        result = json.loads(output_text[start:end])
    except Exception as e:
        print(f"⚠️ Error parsing domain '{domain}': {e}")
        print("Raw output:\n", output_text)
        return None

    return result


# 📌 Load CSV of domains
# Expected columns: 'domain', 'description'
df_domains = pd.read_csv("domains.csv")

# 📌 Evaluate all domains with progress bar
results = []
for idx, row in tqdm(df_domains.iterrows(), total=len(df_domains)):
    res = evaluate_domain(row["domain"], row["description"])
    results.append(res)

# 📌 Combine results into a DataFrame
df_results = pd.DataFrame(results)

# Optional: save results to CSV
df_results.to_csv("domain_scores.csv", index=False)
