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
python src/fine_tune.py
```

Convert to gguf:
```bash
python src/convert_to_gguf.py

git clone https://github.com/ggml-org/llama.cpp.git
python3 llama.cpp/convert_hf_convert_to_gguf.py --outfile llama3_8b_domain.gguf ./llama3_8b_merged
./quantize llama3_8b_domain.gguf llama3_8b_domain.Q4_K_M.gguf Q4_K_M

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