#!/usr/bin/env python
"""Teacher-forced NLL of a long KV-cache-dependent context, per KV dtype.

Why NLL and not PPL: perplexity is a coarse aggregate and on this model the KV
quantisation loss sits near the noise floor of a few-thousand-token PPL window.
Mean negative log-likelihood over ~10^5 teacher-forced tokens has a standard
error small enough to resolve a fraction of a percent, and it is a direct
estimate of the same quantity PPL exponentiates.

The prompt is passed as token ids, not text, so every configuration scores
byte-identical sequences (no tokenizer round-trip drift).

    VLLM_NLL_KV=turboquant_3bit_nc VLLM_NLL_WINDOW=32768 \
    VLLM_NLL_ATTN=TURBOQUANT VLLM_NLL_OUT=/tmp/x.json \
    python harnesses/sm70_kv_nll.py

Env:
  VLLM_NLL_KV      kv_cache_dtype            (default turboquant_3bit_nc)
  VLLM_NLL_ATTN    attention backend         (default TURBOQUANT)
  VLLM_NLL_WINDOW  tokens scored             (default 32768)
  VLLM_NLL_MODEL, VLLM_NLL_CORPUS, VLLM_NLL_TP, VLLM_NLL_UTIL,
  VLLM_NLL_MAX_BT, VLLM_NLL_MAXLEN, VLLM_NLL_OUT
"""

import json
import math
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "vllm"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.getenv(
    "VLLM_NLL_MODEL", "/path/to/model"
)
CORPUS = os.getenv("VLLM_NLL_CORPUS", "/tmp/vllm_sm70/quality_corpus.txt")
KV = os.getenv("VLLM_NLL_KV", "turboquant_3bit_nc")
ATTN = os.getenv("VLLM_NLL_ATTN", "TURBOQUANT")
WINDOW = int(os.getenv("VLLM_NLL_WINDOW", "32768"))
TP = int(os.getenv("VLLM_NLL_TP", "2"))
UTIL = float(os.getenv("VLLM_NLL_UTIL", "0.98"))
MAX_BT = int(os.getenv("VLLM_NLL_MAX_BT", "2048"))
OUT = os.getenv("VLLM_NLL_OUT", "")


def corpus_ids(tok, n):
    """Deterministic first-n token ids of the corpus."""
    text = open(CORPUS, encoding="utf-8", errors="ignore").read()
    ids = tok(text)["input_ids"]
    if len(ids) < n + 8:
        raise SystemExit(f"corpus too short: {len(ids)} < {n}")
    return ids[:n]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
    ids = corpus_ids(tok, WINDOW + 1)
    # Score ids[0:WINDOW]; the model needs one extra token to shift into.
    prompt_ids = ids[:WINDOW]

    maxlen = int(os.getenv("VLLM_NLL_MAXLEN", str(max(WINDOW, 4096))))
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        dtype="float16",
        tensor_parallel_size=TP,
        gpu_memory_utilization=UTIL,
        max_model_len=maxlen,
        max_num_batched_tokens=MAX_BT,
        kv_cache_dtype=KV,
        attention_backend=ATTN,
        mamba_ssm_cache_dtype="float16",
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=True,
    )

    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=0,  # 0 => only the sampled position's own logprob
        ignore_eos=True,
    )
    t0 = time.perf_counter()
    out = llm.generate({"prompt_token_ids": list(prompt_ids)}, params)
    wall = time.perf_counter() - t0

    req = out[0]
    usage = req.prompt_token_ids and len(req.prompt_token_ids)
    plp = req.prompt_logprobs or []
    # Position 0 has no conditioning context (None); everything else is defined.
    nlls = []
    pos_idx = []
    for i, entry in enumerate(plp):
        if not entry:
            continue
        for _tok_id, lp in entry.items():
            v = lp.logprob
            if v is None or not math.isfinite(v):
                continue
            nlls.append(-v)
            pos_idx.append(i)

    if not nlls:
        raise SystemExit("no finite prompt logprobs returned")

    n = len(nlls)
    mean = sum(nlls) / n
    var = sum((x - mean) ** 2 for x in nlls) / max(n - 1, 1)
    se = math.sqrt(var / n)

    # NLL in quarters of the window: does the loss grow with context depth?
    quart = []
    for q in range(4):
        lo, hi = int(n * q / 4), int(n * (q + 1) / 4)
        seg = nlls[lo:hi]
        if seg:
            quart.append(sum(seg) / len(seg))

    # Paired analysis across configs needs the raw per-token NLL vector: the
    # variance of the DIFFERENCE between two KV dtypes is far smaller than the
    # variance of either NLL series, so pairing resolves effects that a
    # comparison of means cannot see at these token counts.
    npy = os.getenv("VLLM_NLL_NPY", "")
    if npy:
        import numpy as np

        np.save(npy, np.asarray(nlls, dtype=np.float64))

    res = {
        "kv_cache_dtype": KV,
        "attention_backend": ATTN,
        "window": WINDOW,
        "scored_tokens": n,
        "prompt_tokens": usage,
        "max_model_len": maxlen,
        "mean_nll": mean,
        "se": se,
        "ppl": math.exp(mean),
        "ppl_lo": math.exp(mean - se),
        "ppl_hi": math.exp(mean + se),
        "nll_quartiles": quart,
        "wall_s": wall,
        "prefill_tok_s": usage / wall if wall else None,
        "distinct_scored": len({round(x, 6) for x in nlls}),
        "finite": bool(all(math.isfinite(x) for x in nlls)),
        "npy": npy or None,
    }
    print(json.dumps(res, indent=2))
    if OUT:
        with open(OUT, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
