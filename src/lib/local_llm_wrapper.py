"""
local_llm_wrapper.py

LangChain-compatible wrapper for running local Hugging Face Transformers models.

───────────────────────────────────────────────────────────────────────────────

Why do we need this?

LangChain supports many LLM providers like OpenAI and Hugging Face Hub (API-based),
but it does not include built-in support for models running locally via the
`transformers` library.

This wrapper bridges that gap by adapting a local Hugging Face Transformers model
to LangChain’s expected interface.

Benefits of using a local model with this wrapper:
- No API keys or internet access required
- Improved privacy: all data stays on your machine
- Zero cost per call: no billing or rate limits
- More flexibility: use any Hugging Face model (e.g. flan, Mistral, GPT-Neo, etc.)

You can plug this into LangChain chains like `RetrievalQA` exactly as you would
an API-based model; without changing your pipeline logic.

Version: First version is from my other project:
https://github.com/ferdinand-koenig/insight-bridge/blob/main/app/inference/local_llm_wrapper.py

Here, we adjusted the logger and call function to work stand-alone and without langchain
"""

import logging
import os
from typing import Any, List, Mapping, Optional

import torch
from langchain.llms.base import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field, PrivateAttr
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, AutoTokenizer,
                          PreTrainedModel, PreTrainedTokenizerBase)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("DomainGenerator")

# Optional GGUF optimization
os.environ["GGUF_USE_MMAP"] = "1"
logger.debug(f"Set GGUF_USE_MMAP to: {os.environ['GGUF_USE_MMAP']}")


class LocalTransformersLLM(LLM):
    """LangChain-compatible wrapper for a local Hugging Face Transformers model.
    Also directly callable as a function (`llm("prompt")`).
    """

    # LangChain / Pydantic fields
    model_name: str = Field(..., description="Hugging Face model repo id")
    gguf_file: Optional[str] = Field(
        default=None,
        description="Optional GGUF filename inside repo (e.g. model.Q4_K_M.gguf)",
    )
    max_length: int = Field(
        default=512, description="Maximum output token length"
    )
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Nucleus sampling top_p")
    do_sample: bool = Field(
        default=True, description="Whether to sample or do greedy decoding"
    )
    num_beams: int = Field(
        default=1, description="Number of beams for beam search"
    )

    # Private attrs (not serialized by LangChain)
    _tokenizer: PreTrainedTokenizerBase = PrivateAttr()
    _model: PreTrainedModel = PrivateAttr()
    _device: torch.device = PrivateAttr()
    _is_seq2seq: bool = PrivateAttr()

    def __init__(
        self, model_name: str, gguf_file: Optional[str] = None, **kwargs: Any
    ):
        super().__init__(model_name=model_name, gguf_file=gguf_file, **kwargs)
        self.model_name = model_name
        self.gguf_file = gguf_file

        logger.debug(f"Loading model: {model_name}, gguf_file={gguf_file}")

        # Load tokenizer
        self._tokenizer: PreTrainedTokenizerBase = (
            AutoTokenizer.from_pretrained(
                self.model_name,
                gguf_file=self.gguf_file,
            )
        )

        # Inspect config to decide Seq2Seq vs. Causal
        config = AutoConfig.from_pretrained(
            self.model_name, gguf_file=self.gguf_file
        )

        if config.is_encoder_decoder:
            self._model: PreTrainedModel = (
                AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    gguf_file=self.gguf_file,
                    torch_dtype=torch.float32,  # GGUF always dequantizes to FP32
                )
            )
            self._is_seq2seq = True
            logger.debug("Loaded as Seq2Seq model")
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                gguf_file=self.gguf_file,
                torch_dtype=torch.float32,
            )
            self._is_seq2seq = False
            logger.debug("Loaded as Causal LM model")

        # Select device
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model.to(self._device)
        logger.info(f"Model loaded on device: {self._device}")

    # ------------------------------------------------------------------
    # LangChain integration: implement _call with correct signature
    # ------------------------------------------------------------------
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self._call_transformers(prompt, stop=stop)

    # ------------------------------------------------------------------
    # Direct use: make the class callable like a plain wrapper
    # ------------------------------------------------------------------
    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self._call_transformers(prompt, **kwargs)

    # ------------------------------------------------------------------
    # Shared generation logic
    # ------------------------------------------------------------------
    def _call_transformers(
        self, prompt: str, stop: Optional[List[str]] = None
    ) -> str:
        logger.debug(f'Prompt passed to tokenizer and model: "{prompt}"')
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        if self._is_seq2seq:
            outputs = self._model.generate(
                **inputs,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                early_stopping=True,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        else:
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                early_stopping=True,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        return self._tokenizer.decode(
            outputs[0], skip_special_tokens=True
        ).strip()

    # ------------------------------------------------------------------
    # Required by LangChain
    # ------------------------------------------------------------------
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "gguf_file": self.gguf_file,
            "max_length": self.max_length,
            "is_seq2seq": self._is_seq2seq,
        }

    @property
    def _llm_type(self) -> str:
        return "transformers_gguf"
