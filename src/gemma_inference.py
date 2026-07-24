"""
gemma_inference.py

Chapter 12.12 -- Gemma 3 Inference Module.

This is the final stage of the RAG pipeline that app.py's /ask endpoint
currently stops short of:

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
    - Load google/gemma-3-12b-it once, lazily, and cache it in memory.
    - Auto-detect CUDA vs CPU.
    - Expose generate_answer(prompt) -> str, taking the exact prompt
      string produced by prompt_engineering.build_prompt() and
      returning Gemma 3's decoded answer text.
    - Provide a small CLI for standalone testing (`python -m
      src.gemma_inference`).

This module does NOT wire itself into app.py and does NOT implement
streaming -- those are explicitly out of scope per this chapter and
are left for a later chapter.

4-bit quantization (bitsandbytes) is now used by default on GPU: it
cuts each loaded copy of Gemma 3 12B from ~24GB down to ~7-8GB of
VRAM, and speeds up generation, with a small, usually-unnoticeable
quality tradeoff for extractive/QA-style answers over retrieved
context. This also matters operationally: if this module ever ends
up loaded twice on the same GPU (e.g. once in a notebook kernel for
testing, once again in a server subprocess started from that same
notebook), two 4-bit copies still comfortably fit on a single 40GB
GPU, whereas two full-precision bf16 copies do not -- and running out
of headroom is exactly what causes silent slowdowns/timeouts on
larger-context (free-text, non-clause) queries. Set
GEMMA_USE_4BIT=0 in the environment to disable and load full
bfloat16 precision instead.
"""

import glob
import logging
import os
import site
from typing import Optional, Tuple

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

try:
    from transformers import BitsAndBytesConfig
    _BITSANDBYTES_AVAILABLE = True
except ImportError:  # bitsandbytes not installed
    _BITSANDBYTES_AVAILABLE = False

logger = logging.getLogger("dmrc_rag.gemma_inference")


def _fixup_nvjitlink_ld_library_path() -> None:
    """Best-effort workaround for a known bitsandbytes/Colab issue
    (bitsandbytes-foundation#1905): recent bitsandbytes builds dlopen
    libnvJitLink.so.13 (a CUDA 13 runtime lib), but it's frequently not
    on the dynamic linker's search path even when the pip package that
    ships it (nvidia-nvjitlink-cu13) is installed, because dlopen'd
    transitive dependencies rely on LD_LIBRARY_PATH rather than the
    package's own location. This searches site-packages for the file
    and, if found, prepends its directory to LD_LIBRARY_PATH *before*
    bitsandbytes is ever imported (import happens lazily, inside
    get_gemma_model() below, the first time 4-bit loading is
    attempted). Purely additive -- if nothing is found, this is a
    no-op, and get_gemma_model() falls back to full precision anyway
    if bitsandbytes still can't load.
    """
    try:
        search_roots = list(site.getsitepackages())
        try:
            search_roots.append(site.getusersitepackages())
        except Exception:
            pass

        found_dirs = set()
        for root in search_roots:
            for pattern in ("libnvJitLink.so.13", "libnvJitLink.so.13.*"):
                found_dirs.update(
                    os.path.dirname(p)
                    for p in glob.glob(os.path.join(root, "**", pattern), recursive=True)
                )

        if not found_dirs:
            return

        current = os.environ.get("LD_LIBRARY_PATH", "")
        new_path = os.pathsep.join([*found_dirs, current]) if current else os.pathsep.join(found_dirs)
        os.environ["LD_LIBRARY_PATH"] = new_path
        logger.info("Added to LD_LIBRARY_PATH for bitsandbytes: %s", found_dirs)
    except Exception:
        # Never let this best-effort fixup break model loading.
        logger.debug("nvJitLink LD_LIBRARY_PATH fixup failed (non-fatal).", exc_info=True)


_fixup_nvjitlink_ld_library_path()


# ---------------------------------------------------------------------------
# Configuration
#
# google/gemma-3-12b-it is Gemma 3's instruction-tuned 12B checkpoint.
# The 4B/12B/27B Gemma 3 checkpoints are vision-language models under
# the hood, so the correct Transformers classes are
# Gemma3ForConditionalGeneration + AutoProcessor (rather than a plain
# AutoModelForCausalLM + AutoTokenizer pair, which is only correct for
# the 1B text-only checkpoint). We only ever feed this module text, so
# in practice it behaves exactly like a text-in/text-out chat model --
# the processor's chat template handles pure-text messages fine.
# ---------------------------------------------------------------------------

MODEL_NAME = "google/gemma-3-12b-it"

# 12.6 Generation defaults. Greedy decoding (do_sample=False) is used
# for reproducible, low-variance answers over contract/engineering
# text, where consistency matters more than creative variation.
# NOTE: temperature has no effect when do_sample=False (greedy
# decoding ignores it); it is kept here, set low, so that flipping
# do_sample to True later (e.g. for exploratory/creative use) already
# has a sensible, conservative value in place.
TEMPERATURE = 0.2
DO_SAMPLE = False
MAX_NEW_TOKENS = 512

# Load in 4-bit (bitsandbytes) on GPU by default -- see module docstring.
# Falls back to full bfloat16 automatically if bitsandbytes isn't
# installed, or if GEMMA_USE_4BIT=0 is set in the environment.
USE_4BIT = _BITSANDBYTES_AVAILABLE and os.environ.get("GEMMA_USE_4BIT", "1") != "0"


# ---------------------------------------------------------------------------
# 12.4 Lazy-loaded, cached model + processor
#
# Mirrors the pattern already used by query.get_model() (Chapter 7/9)
# and reranker.get_reranker_model() (Chapter 10): nothing is loaded at
# import time. The first call to get_gemma_model() pays the (large,
# multi-second-to-multi-minute) load cost; every subsequent call reuses
# the same in-memory model and processor via these module-level
# caches, so a FastAPI warm-up hook (Chapter 14.9) can call this once
# at startup, or the first real request can trigger the load lazily.
# ---------------------------------------------------------------------------

_model: Optional[Gemma3ForConditionalGeneration] = None
_processor: Optional[AutoProcessor] = None
_device: Optional[str] = None


def _select_device() -> str:
    """GPU auto-detection: use CUDA if a GPU is visible to PyTorch,
    otherwise fall back to CPU. Gemma 3 12B is large enough that CPU
    inference will be slow, but it should still work correctly -- this
    module never hard-requires a GPU.
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info("CUDA GPU detected (%s); using device='cuda'.", device_name)
        return "cuda"
    logger.info("No CUDA GPU detected; falling back to device='cpu'.")
    return "cpu"


def get_gemma_model() -> Tuple[Gemma3ForConditionalGeneration, AutoProcessor, str]:
    """Initialize (on first call) and return the cached
    (model, processor, device) triple.

    Lazy loading: the tokenizer/processor and the 12B-parameter model
    weights are only pulled from disk/Hugging Face Hub and placed on
    the GPU/CPU the first time this function is called. Every later
    call -- from generate_answer(), from a FastAPI warm-up hook, or
    from the CLI below -- returns the same cached objects instead of
    reloading them, since reloading a 12B model per-request would be
    far too slow for real-time question answering.
    """
    global _model, _processor, _device

    if _model is not None and _processor is not None and _device is not None:
        return _model, _processor, _device

    _device = _select_device()

    logger.info("Loading Gemma 3 processor: %s", MODEL_NAME)
    _processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # bfloat16 on GPU keeps memory/latency reasonable for a 12B model;
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
        logger.info("Loading Gemma 3 model: %s (4-bit nf4 quantized, device=%s)", MODEL_NAME, _device)
    else:
        logger.info("Loading Gemma 3 model: %s (dtype=%s, device=%s)", MODEL_NAME, torch_dtype, _device)

    try:
        _model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            device_map="auto" if _device == "cuda" else None,
            quantization_config=quantization_config,
        ).eval()
    except Exception:
        if quantization_config is None:
            raise  # not a quantization issue -- a real failure, don't hide it
        # 4-bit loading failed (e.g. a bitsandbytes/CUDA library mismatch
        # like bitsandbytes-foundation#1905 -- libnvJitLink.so.13 not
        # found on this particular Colab GPU/CUDA build). Don't let an
        # environment-specific quantization problem crash the whole
        # notebook: fall back to full-precision bfloat16 instead.
        logger.warning(
            "4-bit quantized load of %s failed; falling back to full bfloat16 "
            "precision. Set GEMMA_USE_4BIT=0 to skip the 4-bit attempt entirely "
            "next time.",
            MODEL_NAME,
            exc_info=True,
        )
        _model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            device_map="auto" if _device == "cuda" else None,
        ).eval()

    # device_map="auto" already places the model correctly when a GPU
    # is present; on CPU there's no device_map, so move explicitly.
    if _device == "cpu":
        _model = _model.to(_device)

    logger.info("Gemma 3 model and processor loaded and cached.")
    return _model, _processor, _device


# ---------------------------------------------------------------------------
# 12.12 Inference -- prompt in, decoded answer string out
# ---------------------------------------------------------------------------

def generate_answer(prompt: str) -> str:
    """Run Gemma 3 inference on a fully-assembled prompt string.

    Args:
        prompt: The complete prompt produced by
            prompt_engineering.build_prompt() (retrieved context +
            question, already formatted). This function does not
            construct or modify the prompt in any way -- it treats it
            as a single opaque user message.

    Returns:
        The generated answer as a plain string, with the input prompt
        and any special tokens stripped out (i.e. only the newly
        generated continuation, decoded).
    """
    model, processor, device = get_gemma_model()

    # The chat template wraps the raw prompt string as a single user
    # turn. Gemma 3's processor understands plain-text-only messages
    # (no "image" content needed) and returns a dict of tensors ready
    # for model.generate(), exactly as it would for a multimodal turn.
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    input_length = inputs["input_ids"].shape[-1]

    # inference_mode disables gradient tracking, which we never need
    # here and which would otherwise waste memory/compute.
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
        )

    # model.generate() returns the full sequence (prompt tokens +
    # newly generated tokens) concatenated together. Slicing off the
    # first `input_length` tokens keeps only what Gemma 3 actually
    # generated, so callers get just the answer -- not their own
    # prompt echoed back to them.
    new_tokens = generation[0][input_length:]

    answer = processor.decode(new_tokens, skip_special_tokens=True)
    return answer.strip()


# ---------------------------------------------------------------------------
# CLI -- standalone manual testing: `python -m src.gemma_inference`
# ---------------------------------------------------------------------------

def _run_cli() -> None:
    logging.basicConfig(level=logging.INFO)

    print(f"Gemma 3 inference CLI -- model: {MODEL_NAME}")
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
