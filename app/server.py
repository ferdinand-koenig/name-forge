import threading
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from src.domain_generator import DomainGenerator
from src.lib.local_llm_wrapper import LocalTransformersLLM


# -------------------------------
# Request / Response Models
# -------------------------------
class GenerateRequest(BaseModel):
    business_description: str


class DomainSuggestion(BaseModel):
    domain: str
    confidence: float


class GenerateResponse(BaseModel):
    suggestions: List[DomainSuggestion]
    status: str
    message: str = None


# -------------------------------
# Initialize LLM and generator
# -------------------------------
MODEL_PATH = "artifacts/mistral_7B_lora-q4_k_m-v2.1.gguf"
llm = LocalTransformersLLM(
    model_name=MODEL_PATH,
    max_length=50,
    do_sample=True,
    temperature=0.7,
)

generator = DomainGenerator(llm=llm)
llm_lock = threading.Lock()
# -------------------------------
# Create FastAPI app
# -------------------------------
app = FastAPI(title="Domain Suggestion API")


@app.get("/")
def redirect_to_github():
    return RedirectResponse("https://github.com/ferdinand-koenig/name-forge")


@app.get("/download")
def download_artifacts():
    return FileResponse(
        "artifacts.zip", media_type="application/zip", filename="artifacts.zip"
    )


@app.post("/generate", response_model=GenerateResponse)
def generate_domains(request: GenerateRequest):
    with llm_lock:
        domains = generator.generate(request.business_description)

    # Safety check
    if "__BLOCKED__" in domains:
        return GenerateResponse(
            suggestions=[],
            status="blocked",
            message="Request contains inappropriate content",
        )

    # For simplicity, assign dummy confidence scores
    suggestions = [
        DomainSuggestion(domain=d, confidence=round(1.0 - i * 0.07, 2))
        for i, d in enumerate(domains)
    ]

    return GenerateResponse(suggestions=suggestions, status="success")


# -------------------------------
# Optional: run standalone
# -------------------------------
if __name__ == "__main__":
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000)
