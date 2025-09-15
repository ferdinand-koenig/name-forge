# NameForge – AI-Powered Domain Name Generator
<a href="https://creativecommons.org/licenses/by-nc-nd/4.0/">
  <img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc-nd.png" width="58"/>
</a>

NameForge is a project that leverages Large Language Models (LLMs) to generate creative, relevant, and safe domain name suggestions from business descriptions. The project emphasizes **dataset creation, model fine-tuning, evaluation, edge case discovery, and iterative improvement**.

- - -

## API Example

### underground techno venue Berlin Mitte
```cmd
curl -X POST "https://llm.koenix.de/domain-generator/generate"      -H "Content-Type: application/json"      -d '{"business_description": "underground techno venue Berlin Mitte"}'
```

```json
{"suggestions":[
  {"domain":"undergroundtechno.com","confidence":1.0},
  {"domain":"berlinclub.net","confidence":0.93},
  {"domain":"techno-party.org","confidence":0.86}],
  "status":"success","message":null}

```

### Berlin-based techno music festival and event space for electronic music lovers
```cmd
curl -X POST "https://llm.koenix.de/domain-generator/generate"      -H "Content-Type: application/json"      -d '{"business_description": "Berlin-based techno music festival and event space for electronic music lovers"}'
```

```json
{"suggestions":[
  {"domain":"berlinfestival.com","confidence":1.0},
  {"domain":"technoevents.net","confidence":0.93},
  {"domain":"electronicmusichub.org","confidence":0.86}],
  "status":"success","message":null}

```


### Blocked Request Example: Sex and erotic club in Berlin
```cmd
curl -X POST "https://llm.koenix.de/domain-generator/generate"      -H "Content-Type: application/json"      -d '{"business_description": "Sex and erotic club in Berlin"}'
```


**Response:**

```json
{
  "suggestions": [],
  "status": "blocked",
  "message": "Request contains inappropriate content"
}
```

- - -

## Project Structure
```
./                              # Root folder
    app/                        # API code and entrypoints
    artifacts/                  # Model files, tokenizer, templates
    data/                       # Datasets and data generation scripts
        llm-as-a-judge-training/ # Sample datasets for LLM evaluation
        raw/                     # Raw train/test CSV datasets
    hooks/                      # Pre-commit hooks
    img/                        # Images for documentation or notebooks
    outputs/                    # Evaluation outputs
        judged/                  # Judged results and summary CSVs
    prompts/                    # YAML prompts for generation/evaluation
    src/                        # Source code
        eval/                    # Evaluation scripts & Dockerfile
        fine_tune/               # Fine-tuning scripts and data utilities
        lib/                     # LLM wrapper
```
- - -

## Setup Instructions

This project has multiple environments depending on the phase of the workflow. Choose the appropriate environment based on your task.  

### 1. Clone the Repository

```bash
git clone https://github.com/ferdinand-koenig/name-forge.git  
cd name-forge
```

### 2. Environment Setup by Task

| Task | Recommended Tool | Notes |
|------|-----------------|-------|
| **Data Generation** | Pipenv | Used for GPU-based tasks and synthetic dataset creation. |
| **Fine-tuning** | Pipenv | Handles GPU-heavy model fine-tuning workflows. |
| **Evaluation / LLM-as-a-Judge** | Pipenv (scripts), Docker (test result generation) | Analysis uses Pipenv; Docker container available in `src/eval/Dockerfile` for generating outputs. |
| **Deployment / Inference** | Docker (CPU, Poetry used internally) | Docker container available at project root `Dockerfile`. |


### 3. Using Pipenv (GPU Tasks / Evaluation Scripts)

```bash
# Install Pipenv if not installed
pip install pipenv

# Activate environment
pipenv shell

# Install dependencies
pipenv install
```

### 4. Using Docker
> **For Unix users:** Replace line break ` with \

- **Evaluation / Test Generation Image:** `eval-name-forge`

```bash
docker run -it --rm `
  -v C:/Users/koenig/PycharmProjects/NameForge/artifacts:/insight-bridge/artifacts `
  -v C:/Users/koenig/PycharmProjects/NameForge/outputs:/insight-bridge/outputs `
  eval-name-forge
```

- **Inference / Deployment Image:** `server-name-forge`

```bash
docker run -it --rm `
  -v C:/Users/koenig/PycharmProjects/NameForge/artifacts:/insight-bridge/artifacts `
  -v pip_cache:/root/.cache/pip `
  -p 8000:8000 `
  server-name-forge
```

[//]: # (## Project Workflow)

[//]: # ()
[//]: # (1.  **Synthetic Dataset Creation**)

[//]: # (    *   Generate diverse business descriptions and corresponding domain names.)

[//]: # (    *   Preprocess and save datasets in `data/`.)

[//]: # (2.  **Model Training**)

[//]: # (    *   Fine-tune a baseline open-source LLM &#40;LoRA or full fine-tuning&#41;.)

[//]: # (    *   Save checkpoints in `checkpoints/`.)

[//]: # (3.  **Evaluation Framework**)

[//]: # (    *   LLM-as-a-judge scoring for relevance, creativity, and safety.)

[//]: # (    *   Store metrics in `experiments/`.)

[//]: # (4.  **Edge Case Discovery & Iterative Improvement**)

[//]: # (    *   Identify failure modes and retrain to improve performance.)

[//]: # (    *   Save improved model checkpoints and updated evaluation metrics.)

[//]: # (5.  **Safety Guardrails**)

[//]: # (    *   Ensure inappropriate or harmful content is blocked.)

[//]: # (6.  **FastAPI Server &#40;Optional&#41;**)

[//]: # (    *   Launch API endpoint in `server/app.py` for production-like usage.)

- - -



## Reproducibility

*   All experiments are reproducible via Pipenv.
*   Datasets, and evaluation results are versioned in the repo.

---

## Downloading the models
The models are too big for Github (Size of ~4GB per quantized model). To download them, visit
[https://llm.koenix.de/domain-generator/download](https://llm.koenix.de/domain-generator/download)





















---

## notes

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pipenv
pipenv install
python src/fine_tune/fine_tune.py
```

or with only devices 0-2:
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python src/fine_tune/fine_tune.py
```

Convert to gguf:
```bash
python src/fine_tune/convert_to_gguf.py

git clone https://github.com/ggml-org/llama.cpp.git
python3 llama.cpp/convert_hf_to_gguf.py --outfile mistral_7B_lora.gguf ./mistral_7B_merged

```
![img.png](img/wizard.png)

Quantized with quantizer from source
```bash
# in llama.cpp repository
wget https://github.com/ggml-org/llama.cpp/releases/download/b6401/llama-b6401-bin-ubuntu-x64.zip
unzip llama-b6401-bin-ubuntu-x64.zip
cd ..
# back again in NameForge (project root)
llama.cpp/release/build/bin/llama-quantize ./mistral_7B_lora.gguf ./mistral_7B_lora-q4_k_m.gguf Q4_K_M
```


test

```bash
llama.cpp/release/build/bin/llama-cli -m ./mistral_7B_lora-q4_k_m.gguf -p 'You are a domain name generator AI. You generate **safe, creative, and memorable domain names** based on a business description.
  - Do not generate inappropriate content.
  - Use only alphanumeric characters and standard TLDs (.com, .net, .org, etc.).
  - Provide 2–3 suggestions per description.
  - If the description is unsafe, output exactly: "__BLOCKED__".

  --- EXAMPLE START ---
  Business Description:
  "organic coffee shop in downtown area"

  Output:
  ["organicbeanscafe.com", "downtowncoffee.org", "freshbreworganic.net"]
  --- EXAMPLE END ---

  --- TASK START ---
  Business Description:
  A local farmer offering fresh eggs

  Output only a JSON array of 2–3 domain names:'
```

```bash
llama.cpp/release/build/bin/llama-cli -m ./mistral_7B_lora-q4_k_m.gguf --max-tokens 50 --stop "--- TASK END" -p 'You are a domain name generator AI. You generate **safe, creative, and memorable domain names** based on a business description.
  - Do not generate inappropriate content.
  - Use only alphanumeric characters and standard TLDs (.com, .net, .org, etc.).
  - Provide 2–3 suggestions per description.
  - If the description is unsafe, output exactly: "__BLOCKED__".

  --- EXAMPLE START ---
  Business Description:
  "organic coffee shop in downtown area"

  Output:
  ["organicbeanscafe.com", "downtowncoffee.org", "freshbreworganic.net"]
  --- EXAMPLE END ---

  --- TASK START ---
  Business Description:
  adult website

  Output only a JSON array of 2–3 domain names:'
  ```




Command for evaluation:
python .\src\eval\simple_judge.py
python .\src\eval\get_pearsons_corr.py
