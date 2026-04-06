"""
DiffusionLLM API Server

Serves Dream-v0-Instruct-7B and/or LLaDA-8B-Instruct via a FastAPI server.
Exposes /generate and /tokens endpoints as expected by api_dllm.py.

Usage:
    # Serve Dream only (default):
    python dllm_server.py --model dream --port 23450

    # Serve LLaDA only:
    python dllm_server.py --model llada --port 23450

Environment:
    Use dllm conda env:
    /home/u7444045/miniconda3/envs/dllm/bin/python dllm_server.py ...
"""

import argparse
import os
import sys
import logging
import time
import torch
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    messages: List[Message]
    model: str = "Dream"
    gen_length: int = 128
    steps: int = 128
    temperature: float = 0.0
    dual_cache: bool = True
    block_size: int = 32
    threshold: float = 0.9
    return_tokens: bool = False

class GenerateResponse(BaseModel):
    response: str
    token: int

class TokenizeRequest(BaseModel):
    model: str
    messages: List[Message]

class TokenizeResponse(BaseModel):
    num_of_tokens: int

# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_model_name = None  # "Dream" or "LLaDA"

DREAM_MODEL_PATH = "/data/model_hub/hf/Dream-v0-Instruct-7B"
LLADA_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
LLADA_CACHE_DIR = "/data/.cache/huggingface/hub"


def _patch_multinomial():
    """Patch torch.multinomial to handle numerical issues (from Dream codebase)."""
    import torch

    def _sanitize(probs):
        if not torch.is_floating_point(probs):
            return probs
        p = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        if p.ndim == 1:
            t = p.sum()
            return p / t if t > 0 else torch.full_like(p, 1.0 / p.numel())
        t = p.sum(dim=-1, keepdim=True)
        zero = t <= 0
        if zero.any():
            p = p.clone()
            p[zero.squeeze(-1)] = 1.0
            t = p.sum(dim=-1, keepdim=True)
        return p / t

    _orig = torch.multinomial

    def _safe(probs, num_samples, replacement=False, *, generator=None, out=None):
        return _orig(_sanitize(probs), num_samples, replacement, generator=generator, out=out)

    if not getattr(torch.multinomial, "_dllm_safe", False):
        _safe._dllm_safe = True
        torch.multinomial = _safe


def load_model(model_type: str):
    global _model, _tokenizer, _model_name
    from transformers import AutoModel, AutoTokenizer

    _patch_multinomial()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # SGLang (Qwen3/Ministral fp8) uses ~14 GB, leaving ~10.5 GB free on a 24 GB GPU.
    # 8-bit quantization reduces LLaDA/Dream from ~16 GB to ~8 GB so both servers fit.
    # device_map="cuda:0" forces all layers onto the GPU (no CPU offload).
    if device == "cuda":
        load_kwargs = dict(load_in_8bit=True, device_map="cuda:0")
    else:
        load_kwargs = dict(torch_dtype=torch.float32)

    if model_type.lower() in ("dream", "dream-v0-instruct-7b"):
        model_path = DREAM_MODEL_PATH
        if not os.path.exists(model_path):
            # Fallback to HuggingFace download
            model_path = "Dream-org/Dream-v0-Instruct-7B"
            logger.info(f"Local Dream path not found, using HuggingFace: {model_path}")
        logger.info(f"Loading Dream model from {model_path} (8-bit) ...")
        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, **load_kwargs
        ).eval()
        _model_name = "Dream"

    elif model_type.lower() in ("llada", "llada-8b-instruct"):
        logger.info(f"Loading LLaDA model (8-bit) ...")
        _tokenizer = AutoTokenizer.from_pretrained(
            LLADA_MODEL_ID, trust_remote_code=True, cache_dir=LLADA_CACHE_DIR
        )
        _model = AutoModel.from_pretrained(
            LLADA_MODEL_ID, trust_remote_code=True,
            cache_dir=LLADA_CACHE_DIR, **load_kwargs
        ).eval()
        _model_name = "LLaDA"

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Model loaded: {_model_name} on {device}")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _apply_chat_template(messages: List[Message]) -> torch.Tensor:
    """Convert messages to input_ids using the tokenizer's chat template."""
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
    # Add generation prompt
    input_ids = _tokenizer.apply_chat_template(
        msg_dicts,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    device = next(_model.parameters()).device
    return input_ids.to(device)


def _generate_dream(input_ids: torch.Tensor, gen_length: int, steps: int, temperature: float) -> str:
    """Run Dream diffusion_generate."""
    with torch.no_grad():
        output = _model.diffusion_generate(
            input_ids,
            max_new_tokens=gen_length,
            output_history=False,
            return_dict_in_generate=True,
            steps=steps,
            temperature=max(temperature, 0.01),  # Dream needs > 0
            alg="entropy",
        )
    new_tokens = output.sequences[0][input_ids.shape[1]:]
    text = _tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
    return text.strip()


def _generate_llada(input_ids: torch.Tensor, gen_length: int, steps: int, temperature: float) -> str:
    """Run LLaDA masked diffusion generation."""
    import torch.nn.functional as F

    MASK_ID = _tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
    prompt_len = input_ids.shape[1]
    device = input_ids.device

    # Initialize masked sequence
    x = torch.full((1, prompt_len + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    with torch.no_grad():
        for i in range(steps):
            t = 1.0 - i / steps
            # How many masks remain
            mask_positions = (x[:, prompt_len:] == MASK_ID).nonzero(as_tuple=False)
            if len(mask_positions) == 0:
                break

            logits = _model(x).logits
            gen_logits = logits[:, prompt_len:, :]  # (1, gen_length, vocab)

            if temperature > 0:
                probs = F.softmax(gen_logits / temperature, dim=-1)
                preds = torch.multinomial(probs.squeeze(0), 1).squeeze(-1)
            else:
                preds = gen_logits.argmax(dim=-1).squeeze(0)

            # Unmask a fraction proportional to step progress
            frac = (i + 1) / steps
            num_to_unmask = max(1, int(frac * gen_length) - int((i / steps) * gen_length))
            mask_idx = (x[0, prompt_len:] == MASK_ID).nonzero(as_tuple=False).squeeze(-1)
            if len(mask_idx) > 0:
                chosen = mask_idx[:num_to_unmask]
                x[0, prompt_len + chosen] = preds[chosen]

    generated = x[0, prompt_len:].tolist()
    text = _tokenizer.decode(
        [t for t in generated if t != MASK_ID],
        skip_special_tokens=True
    )
    return text.strip()


def run_inference(request: GenerateRequest) -> tuple[str, int]:
    """Dispatch inference to the correct model."""
    input_ids = _apply_chat_template(request.messages)
    start = time.time()

    if _model_name == "Dream":
        text = _generate_dream(
            input_ids, request.gen_length, request.steps, request.temperature
        )
    elif _model_name == "LLaDA":
        text = _generate_llada(
            input_ids, request.gen_length, request.steps, request.temperature
        )
    else:
        raise RuntimeError("Model not loaded")

    elapsed = time.time() - start
    # Approximate token count from output
    out_ids = _tokenizer.encode(text, add_special_tokens=False)
    num_tokens = len(out_ids)
    logger.debug(f"Generated {num_tokens} tokens in {elapsed:.2f}s ({num_tokens/elapsed:.1f} tok/s)")
    return text, num_tokens


def count_tokens(messages: List[Message]) -> int:
    """Count tokens in messages."""
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
    input_ids = _tokenizer.apply_chat_template(
        msg_dicts, add_generation_prompt=False, return_tensors="pt"
    )
    return input_ids.shape[1]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="DiffusionLLM Server")


@app.get("/health")
def health():
    return {"status": "ok", "model": _model_name}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        text, num_tokens = run_inference(request)
        return GenerateResponse(response=text, token=num_tokens)
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tokens", response_model=TokenizeResponse)
def tokenize(request: TokenizeRequest):
    if _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        n = count_tokens(request.messages)
        return TokenizeResponse(num_of_tokens=n)
    except Exception as e:
        logger.exception("Tokenization failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffusionLLM API Server")
    parser.add_argument("--model", type=str, default="dream",
                        choices=["dream", "llada"],
                        help="Which DiffusionLLM to serve")
    parser.add_argument("--port", type=int, default=23450,
                        help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    args = parser.parse_args()

    load_model(args.model)
    uvicorn.run(app, host=args.host, port=args.port)
