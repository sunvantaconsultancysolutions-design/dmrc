"""
gemma_inference.py

Chapter 12.12 -- Gemma 2 9B Inference Module.

This is the final stage of the RAG pipeline that app.py's /ask endpoint
wires together:

    User Query
        |
        v
    hybrid_retriever.hybrid_search()   (Chapter 9  -- dense + BM25 + merge)
        |
        v
    reranker.rerank()                  (Chapter 10 -- BGE cross-encoder)
        |
        v
    prompt_engineering.build_prompt()  (Chapter 11 -- prompt assembly)
        |
        v
    gemma_inference.generate_answer()  (Chapter 12.12 -- THIS MODULE)
        |
        v
    JSON response

Scope of this module only:
    - Load google/gemma-2-9b-it once, lazily, and cache it in memory.
    - Auto-detect CUDA vs CPU.
    - Expose generate_answer(prompt) -> str, taking the exact prompt
      string produced by prompt_engineering.build_prompt() and
      returning the model's decoded answer text.
    - Provide a small CLI for standalone testing (`python -m
      src.gemma_inference`).

This module does NOT wire itself into app.py (app.py imports it) and
does NOT implement streaming -- both out of scope here.

MODEL CHANGE (from Gemma 3 12B -> Gemma 2 9B): the task is extractive
QA over retrieved contract clauses -- short, grounded context, low
need for creative generation -- which is exactly where a 7-9B
instruct model performs close to a 12B one. Gemma 2 9B is also a
DIFFERENT ARCHITECTURE from Gemma 3, not just a smaller checkpoint:
    - Gemma 3's 4B/12B/27B checkpoints are vision-language models,
      loaded with Gemma3ForConditionalGeneration + AutoProcessor.
    - Gemma 2 is a plain text-in/text-out causal LM, loaded with the
      standard AutoModelForCausalLM + AutoTokenizer pair, and its
      chat template takes plain string message content (not the
      list-of-typed-parts content Gemma 3's processor expects).
Both call sites below were changed accordingly; the public functions
(get_gemma_model, generate_answer) keep their original names so
app.py's existing imports don't need to change.

4-bit quantization (bitsandbytes) is OFF by default -- see USE_4BIT
below. NF4 dequantizes every weight block back to bf16 before each
matmul; that overhead is invisible on a short clause prompt but
crippling on a long free-text prompt, which is exactly the timeout
this module was patched to fix. Set GEMMA_USE_4BIT=1 in the
environment to opt back in on a smaller GPU.

Gemma 2 also has a known generation quirk on some transformers/PyTorch
SDPA combinations (repeated/garbled tokens with the default attention
implementation). We pin attn_implementation="eager" to sidestep it --
slightly slower than SDPA/flash-attn but numerically reliable, which
matters more for a QA tool that has to be trusted.
"""

import logging
import os
import time
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
    _BITSANDBYTES_AVAILABLE = True
except ImportError:  # bitsandbytes not installed
    _BITSANDBYTES_AVAILABLE = False

logger = logging.getLogger("dmrc_rag.gemma_inference")


# ---------------------------------------------------------------------------
# Configuration
#
# google/gemma-2-9b-it is Gemma 2's instruction-tuned 9B checkpoint --
# a text-only causal LM. Correct Transformers classes are
# AutoModelForCausalLM + AutoTokenizer (NOT Gemma3ForConditionalGeneration
# + AutoProcessor, which is Gemma 3-specific and will fail to load this
# checkpoint correctly).
# ---------------------------------------------------------------------------

MODEL_NAME = "google/gemma-2-9b-it"

# 12.6 Generation defaults. Greedy decoding (do_sample=False) is used
# for reproducible, low-variance answers over contract/engineering
# text, where consistency matters more than creative variation.
# NOTE: temperature has no effect when do_sample=False (greedy
# decoding ignores it); it is kept here, set low, so that flipping
# do_sample to True later (e.g. for exploratory/creative use) already
# has a sensible, conservative value in place.
TEMPERATURE = 0.2
DO_SAMPLE = False
# Generation is sequential, so latency scales linearly with this.
MAX_NEW_TOKENS = int(os.environ.get("GEMMA_MAX_NEW_TOKENS", "320"))

# Load in 4-bit (bitsandbytes) on GPU only if explicitly requested.
# Falls back to full bfloat16 automatically if bitsandbytes isn't
# installed, or if GEMMA_USE_4BIT=0 (the default) is in effect.
USE_4BIT = _BITSANDBYTES_AVAILABLE and os.environ.get("GEMMA_USE_4BIT", "0") == "1"


# ---------------------------------------------------------------------------
# 12.4 Lazy-loaded, cached model + tokenizer
#
# Mirrors the pattern already used by query.get_model() (Chapter 7/9)
# and reranker.get_reranker_model() (Chapter 10): nothing is loaded at
# import time. The first call to get_gemma_model() pays the (large,
# multi-second-to-multi-minute) load cost; every subsequent call reuses
# the same in-memory model and tokenizer via these module-level
# caches, so a FastAPI warm-up hook (Chapter 14.9) can call this once
# at startup, or the first real request can trigger the load lazily.
# ---------------------------------------------------------------------------

_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None
_device: Optional[str] = None


def _select_device() -> str:
    """GPU auto-detection: use CUDA if a GPU is visible to PyTorch,
    otherwise fall back to CPU. Gemma 2 9B is large enough that CPU
    inference will be slow, but it should still work correctly -- this
    module never hard-requires a GPU.
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info("CUDA GPU detected (%s); using device='cuda'.", device_name)
        return "cuda"
    logger.info("No CUDA GPU detected; falling back to device='cpu'.")
    return "cpu"


def get_gemma_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """Initialize (on first call) and return the cached
    (model, tokenizer, device) triple.

    Lazy loading: the tokenizer and the 9B-parameter model weights are
    only pulled from disk/Hugging Face Hub and placed on the GPU/CPU
    the first time this function is called. Every later call -- from
    generate_answer(), from a FastAPI warm-up hook, or from the CLI
    below -- returns the same cached objects instead of reloading
    them, since reloading a 9B model per-request would be far too slow
    for real-time question answering.
    """
    global _model, _tokenizer, _device

    if _model is not None and _tokenizer is not None and _device is not None:
        return _model, _tokenizer, _device

    _device = _select_device()

    logger.info("Loading Gemma 2 tokenizer: %s", MODEL_NAME)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # bfloat16 on GPU keeps memory/latency reasonable for a 9B model;
    # on CPU we let PyTorch pick a safe default float dtype instead,
    # since bfloat16 CPU kernels are inconsistently supported.
    torch_dtype = torch.bfloat16 if _device == "cuda" else torch.float32

    quantization_config = None
    if _device == "cuda" and USE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Loading Gemma 2 model: %s (4-bit nf4 quantized, device=%s)", MODEL_NAME, _device)
    else:
        logger.info("Loading Gemma 2 model: %s (dtype=%s, device=%s)", MODEL_NAME, torch_dtype, _device)

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map="auto" if _device == "cuda" else None,
        quantization_config=quantization_config,
        # BUGFIX (Gemma 2-specific): SDPA/flash-attn have produced
        # garbled/repeated output for Gemma 2 on some transformers +
        # PyTorch combinations. Eager attention is the reliable choice
        # for this model family -- slightly slower, correct output.
        attn_implementation="eager",
    ).eval()

    # device_map="auto" already places the model correctly when a GPU
    # is present; on CPU there's no device_map, so move explicitly.
    if _device == "cpu":
        _model = _model.to(_device)

    logger.info("Gemma 2 model and tokenizer loaded and cached.")
    return _model, _tokenizer, _device


# ---------------------------------------------------------------------------
# 12.12 Inference -- prompt in, decoded answer string out
# ---------------------------------------------------------------------------

def generate_answer(prompt: str, max_new_tokens: Optional[int] = None) -> str:
    """Run Gemma 2 inference on a fully-assembled prompt string.

    Args:
        prompt: The complete prompt produced by
            prompt_engineering.build_prompt() (retrieved context +
            question, already formatted). This function does not
            construct or modify the prompt in any way -- it treats it
            as a single opaque user message.
        max_new_tokens: Optional override of MAX_NEW_TOKENS for this
            call only (e.g. a shorter cap for a warm-up ping).

    Returns:
        The generated answer as a plain string, with the input prompt
        and any special tokens stripped out (i.e. only the newly
        generated continuation, decoded).
    """
    model, tokenizer, device = get_gemma_model()

    # Gemma 2's chat template takes plain string message content
    # (unlike Gemma 3's processor, which expects a list of typed
    # {"type": "text", "text": ...} parts).
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_length = inputs["input_ids"].shape[-1]
    n_new = max_new_tokens if max_new_tokens is not None else MAX_NEW_TOKENS

    # HARD GUARD: refuse to start an enormous prefill silently. Attention cost
    # during prefill grows quadratically with prompt length, so a 10k-token
    # prompt is not 20x a 500-token one -- it is far worse. If this fires, the
    # retrieval layer is handing over too many chunks; fix it there, not here.
    if input_length > 6000:
        logger.warning(
            "Prompt is %d tokens -- very large. Generation will be slow. "
            "Reduce the number of retrieved chunks.", input_length
        )

    _t0 = time.perf_counter()
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=n_new,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE if DO_SAMPLE else None,
            use_cache=True,   # KV cache: without it every token re-attends over
                              # the whole prompt -- O(n^2) and fatal on long context
        )
    _elapsed = time.perf_counter() - _t0
    _gen = generation.shape[-1] - input_length
    print(f"[gemma2] {input_length} prompt tok -> {_gen} new tok in {_elapsed:.1f}s "
          f"({_gen/_elapsed if _elapsed else 0:.1f} tok/s)", flush=True)

    # model.generate() returns the full sequence (prompt tokens +
    # newly generated tokens) concatenated together. Slicing off the
    # first `input_length` tokens keeps only what Gemma 2 actually
    # generated, so callers get just the answer -- not their own
    # prompt echoed back to them.
    new_tokens = generation[0][input_length:]

    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer.strip()


# ---------------------------------------------------------------------------
# CLI -- standalone manual testing: `python -m src.gemma_inference`
# ---------------------------------------------------------------------------

def _run_cli() -> None:
    logging.basicConfig(level=logging.INFO)

    print(f"Gemma 2 inference CLI -- model: {MODEL_NAME}")
    print("Loading model (this can take a while on first run)...")
    get_gemma_model()  # warm up once, up front, so the prompt below isn't the first (slow) call
    print("Model loaded. Type a prompt and press Enter. Ctrl+C to exit.\n")

    try:
        while True:
            prompt = input("Prompt> ").strip()
            if not prompt:
                continue
            answer = generate_answer(prompt)
            print(f"\nAnswer:\n{answer}\n")
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    _run_cli()
