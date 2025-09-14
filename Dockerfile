########################################
# Builder stage: build llama-cpp-python only
########################################
#FROM ghcr.io/ggml-org/llama.cpp:light AS builder
FROM ghcr.io/ggml-org/llama.cpp@sha256:c19f1e324e0e16806eb6e4b9d6fd5081d48018defb8f9a8871fcc1cd867c0a5c AS builder


ARG PYTHON_VERSION=3.10

# 1. Install Python and build tools
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

# 2. Create virtualenv
WORKDIR /insight-bridge
RUN python${PYTHON_VERSION} -m venv .venv
ENV VIRTUAL_ENV=/insight-bridge/.venv \
    PATH="/insight-bridge/.venv/bin:$PATH"

# 3. Install llama-cpp-python from source (with OpenBLAS backend)
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=off -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_NATIVE=ON -DLLAMA_BUILD_BACKEND=ON"

RUN pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python llama-cpp-python==0.3.14


# 4. Export requirements.txt for runtime installation
COPY pyproject.toml ./
RUN pip3 install --no-cache-dir poetry==2.1.2 && \
    poetry self add poetry-plugin-export && \
    poetry export -f requirements.txt --output requirements.txt --without-hashes && \
    # Remove llama-cpp-python line from requirements.txt
    sed -i '/llama-cpp-python/d' requirements.txt


########################################
# Runtime stage: minimal runtime + llama-cpp-python
########################################
#FROM ghcr.io/ggml-org/llama.cpp:light AS runtime
FROM ghcr.io/ggml-org/llama.cpp@sha256:c19f1e324e0e16806eb6e4b9d6fd5081d48018defb8f9a8871fcc1cd867c0a5c AS runtime

ARG PYTHON_VERSION=3.10

# 1. Install Python runtime and OpenBLAS runtime lib
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils \
        python3-pip \
        libopenblas-base \
        ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /insight-bridge

# 2. Copy prebuilt venv with llama-cpp-python
COPY --from=builder /insight-bridge/.venv /insight-bridge/.venv
COPY --from=builder /insight-bridge/requirements.txt /insight-bridge/requirements.txt


ENV VIRTUAL_ENV=/insight-bridge/.venv \
    PATH="/insight-bridge/.venv/bin:$PATH" \
    LD_LIBRARY_PATH=/insight-bridge/.venv/lib/python3.10/site-packages/llama_cpp:/app:${LD_LIBRARY_PATH:-}

# 3. Copy your app code
COPY README.md ./
COPY prompts/prompt-1.yaml ./prompts/
COPY src ./src/
COPY ./app ./app/

# 4. Link llama backend .so files from llama.cpp base image
RUN ln -s /app/libllama.so /insight-bridge/libllama.so && \
    for file in /app/libggml-*.so; do \
        ln -s "$file" /insight-bridge/"$(basename "$file")"; \
    done

# 5. Expose API
EXPOSE 8000

# Make the script executable
RUN chmod +x ./app/entrypoint.sh

ENTRYPOINT ["./app/entrypoint.sh"]
CMD ["python", "-m", "app.server"]
