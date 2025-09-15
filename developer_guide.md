# Developer Guide: Model Fine-Tuning, Conversion, and Testing

This guide covers the steps to set up a development environment, fine-tune the model, convert it to GGUF, quantize it, and run tests using the NameForge project.

---

## 1. Setup Python Environment

Create a virtual environment and install the dependencies:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # Linux / macOS
# .venv\Scripts\activate    # Windows

# Upgrade pip
pip install --upgrade pip

# Install pipenv for managing dependencies
pip install pipenv

# Install all project dependencies using pipenv
pipenv install
```

---

## 2. Fine-Tune the Model

Run the fine-tuning script. By default, it will use all available GPUs:

```bash
pipenv run python src/fine_tune/fine_tune.py
```

To restrict fine-tuning to specific devices (e.g., GPUs 0-2):

```bash
CUDA_VISIBLE_DEVICES=0,1,2 pipenv run python src/fine_tune/fine_tune.py
```

---

## 3. Convert the Model to GGUF Format

After fine-tuning, convert the model to GGUF format for inference:

```bash
pipenv run python src/fine_tune/convert_to_gguf.py

# Clone llama.cpp repository (required for conversion)
git clone https://github.com/ggml-org/llama.cpp.git

# Convert HuggingFace model to GGUF
python3 llama.cpp/convert_hf_to_gguf.py --outfile mistral_7B_lora.gguf ./mistral_7B_merged
```

---

## 4. Quantize the Model

Quantization reduces model size and speeds up inference. Use the llama.cpp quantizer:

```bash
# Download prebuilt quantizer binaries (Linux example)
wget https://github.com/ggml-org/llama.cpp/releases/download/b6401/llama-b6401-bin-ubuntu-x64.zip
unzip llama-b6401-bin-ubuntu-x64.zip
cd ..

# Quantize the GGUF model
llama.cpp/release/build/bin/llama-quantize ./mistral_7B_lora.gguf ./mistral_7B_lora-q4_k_m.gguf Q4_K_M
```

---

## 5. Test the Model

You can test the quantized model using `llama-cli`:

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

For stopping at a specific token or task end:

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

---

## 6. Evaluate Model Outputs

Run the evaluation scripts to assess model performance:

```bash
# Simple scoring
pipenv run python src/eval/simple_judge.py

# Correlation analysis
pipenv run python src/eval/get_pearsons_corr.py
```

---

## Notes

- Always activate the virtual environment (`.venv`) before running Python scripts.
- Use `pipenv run` to ensure the commands execute within the pipenv environment.
- Follow GPU device restrictions if you have limited resources.
- The workflow assumes you have `llama.cpp` cloned for conversion and quantization steps.
