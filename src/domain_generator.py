# domain_generator.py

import json
from typing import List

import yaml

from lib.local_llm_wrapper import LocalTransformersLLM

# -----------------------------
# Load domain prompt from YAML
# -----------------------------
with open("prompts/prompt-1.yaml", "r") as f:
    yaml_content = yaml.safe_load(f)

DOMAIN_PROMPT_TEMPLATE = yaml_content["domain_prompt"]

# -----------------------------
# Load GGUF model via local wrapper
# -----------------------------
MODEL_PATH = "path/to/mistral.gguf"  # replace with your GGUF model
llm = LocalTransformersLLM(
    model_name=MODEL_PATH,
    max_length=50,
    do_sample=True,  # for creative diversity
    temperature=0.7,  # adjust for creativity
)


# -----------------------------
# Domain generation function
# -----------------------------
def generate_domains(desc: str) -> List[str]:
    """
    Generate 2–3 domain names based on a business description.

    Args:
        desc (str): Description of the business/project

    Returns:
        List[str]: List of generated domain names, or ["__BLOCKED__"] if unsafe
    """
    prompt = DOMAIN_PROMPT_TEMPLATE.format(description=desc)
    output = llm(prompt)  # wrapper handles CPU inference

    # Extract JSON array from output
    try:
        start = output.find("[")
        end = output.rfind("]") + 1
        domains = json.loads(output[start:end])
        if isinstance(domains, list):
            return domains
        else:
            print("⚠️ Output is not a list. Raw output:", output)
            return []
    except Exception:
        print("⚠️ Failed to parse JSON. Raw output:", output)
        return []


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    description = "A sustainable eco-friendly packaging company"
    generated_domains = generate_domains(description)
    print("Generated domains:", generated_domains)
