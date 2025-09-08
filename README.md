# NameForge – AI-Powered Domain Name Generator
<a href="https://creativecommons.org/licenses/by-nc-nd/4.0/">
  <img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc-nd.png" width="58"/>
</a>

NameForge is a project that leverages Large Language Models (LLMs) to generate creative, relevant, and safe domain name suggestions from business descriptions. The project emphasizes **dataset creation, model fine-tuning, evaluation, edge case discovery, and iterative improvement**.

- - -

## Project Structure
```
NameForge/
├── data/                # Synthetic datasets
├── notebooks/           # Jupyter notebook with all experiments
├── src/                 # Helper scripts for dataset, model, evaluation, safety
├── checkpoints/         # Saved model checkpoints
├── experiments/         # Evaluation results and metrics
├── reports/             # Technical report
├── server/              # FastAPI server for domain name generation
├── Pipfile              # Pipenv dependency manager
├── Pipfile.lock
└── README.md
```
- - -

## Setup Instructions

1.  **Clone the repo**
    
    ```
    git clone <your-repo-url>
    cd NameForge
    ```
    
2.  **Install dependencies using Pipenv**
    
    ```
    pip install pipenv
    pipenv install
    pipenv shell
    ```
    
3.  **Run Jupyter Notebook**
    
    ```
    jupyter notebook notebooks/NameForge.ipynb
    ```
    

- - -

## Project Workflow

1.  **Synthetic Dataset Creation**
    *   Generate diverse business descriptions and corresponding domain names.
    *   Preprocess and save datasets in `data/`.
2.  **Model Training**
    *   Fine-tune a baseline open-source LLM (LoRA or full fine-tuning).
    *   Save checkpoints in `checkpoints/`.
3.  **Evaluation Framework**
    *   LLM-as-a-judge scoring for relevance, creativity, and safety.
    *   Store metrics in `experiments/`.
4.  **Edge Case Discovery & Iterative Improvement**
    *   Identify failure modes and retrain to improve performance.
    *   Save improved model checkpoints and updated evaluation metrics.
5.  **Safety Guardrails**
    *   Ensure inappropriate or harmful content is blocked.
6.  **FastAPI Server (Optional)**
    *   Launch API endpoint in `server/app.py` for production-like usage.

- - -

## API Example

**Request:**

```
{
  "business_description": "organic coffee shop in downtown area"
}
```

**Response:**

```
{
  "suggestions": [
    {"domain": "organicbeanscafe.com", "confidence": 0.92},
    {"domain": "downtowncoffee.org", "confidence": 0.87}
  ],
  "status": "success"
}
```

**Blocked Request Example:**

```
{
  "business_description": "adult content website with explicit nude content"
}
```

**Response:**

```
{
  "suggestions": [],
  "status": "blocked",
  "message": "Request contains inappropriate content"
}
```

- - -

## Reproducibility

*   All experiments are reproducible via Pipenv.
*   Model checkpoints, datasets, and evaluation results are versioned in the repo.


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



LLM as a judge
1. link all the libs
```bash
for f in ~/NameForge/llama.cpp/release/build/bin/*.so; do
    ln -s "$f" .
done
```
