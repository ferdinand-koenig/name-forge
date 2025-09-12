########################################
# Builder Worker stage: installs Python deps in virtualenv
########################################
FROM ghcr.io/ggml-org/llama.cpp:light AS builder

ARG PYTHON_VERSION=3.10

# 1. Install Python and pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils \
        python3-pip \
        build-essential \
        cmake \
        python3-dev \
        libopenblas-dev \
        pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Install pinned Poetry version
RUN pip3 install poetry==2.1.2

# 3. Configure Poetry environment (create in-project virtualenv) and enable OpenBLAS for llama-cpp-python
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_NO_BINARY="llama-cpp-python" \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    CMAKE_ARGS="-DLLAMA_CUBLAS=off -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_NATIVE=ON -DLLAMA_BUILD_BACKEND=ON"


WORKDIR /name-forge

# 4. Copy dependency files and dummy README for Poetry sanity
COPY pyproject.toml ./
COPY README.md .

# 5. Install all dependencies, forcing llama-cpp-python to build from source
# RUN poetry run pip install --no-binary llama-cpp-python llama-cpp-python && \
RUN poetry install --no-root && \
    rm -rf $POETRY_CACHE_DIR


# Copy code, prompts, dataset, artifacts
COPY data/test_dataset.csv ./data/
COPY prompts/prompt-1.yaml ./prompts/
COPY src ./src/

# 9. Set up environment for virtualenv and add path for llama lib
ENV VIRTUAL_ENV=/name-forge/.venv \
    PATH="/name-forge/.venv/bin:$PATH" \
    LD_LIBRARY_PATH=/name-forge/.venv/lib/python3.11/site-packages/llama_cpp:/app:${LD_LIBRARY_PATH:-}

# Link libllama.so
RUN ln -s /app/libllama.so /name-forge/libllama.so && \
    for file in /app/libggml-*.so; do \
        ln -s "$file" /name-forge/"$(basename "$file")"; \
    done

RUN mkdir -p /name-forge/.venv/lib/python3.11/site-packages/llama_cpp/lib && \
    ln -sf /app/libllama.so /name-forge/.venv/lib/python3.11/site-packages/llama_cpp/lib/libllama.so && \
    for file in /app/libggml-*.so; do \
        ln -sf "$file" /name-forge/.venv/lib/python3.11/site-packages/llama_cpp/lib/; \
    done

# Keep your ENTRYPOINT intact
ENTRYPOINT ["sh", "-c", "\
for model in artifacts/*.gguf; do \
    model_name=$(basename \"$model\" .gguf); \
    echo \"Running domain generation for model: $model_name\"; \
    poetry run python3 -m src.generate_domains_from_csv \
        --input_csv ./data/test_dataset.csv \
        --output_csv ./outputs/${model_name}_domains.csv \
        --model_path \"$model\" \
        --max_length 50 \
        --temperature 0.7; \
done \
"]
