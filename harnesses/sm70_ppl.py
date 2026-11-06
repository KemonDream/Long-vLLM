"""Measure perplexity on a fixed text so KV-quant quality is a number, not a
comment inherited from `turboquant/config.py`.

Uses prompt_logprobs: the engine scores each prompt token under the model, so
mean negative log-likelihood over tokens[1:] gives exp() == perplexity for that
text. Same text, same token window, one run per kv-cache preset.

Usage:
  VLLM_PPL_MODEL=/path/to/model VLLM_PPL_TEXT=/path/text.txt \
  VLLM_PPL_KV_DT=turboquant_4bit_nc VLLM_PPL_TP=2 ... ./venv/bin/python harnesses/sm70_ppl.py
"""

import math
import os


def main() -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = os.environ["VLLM_PPL_MODEL"]
    text_path = os.environ["VLLM_PPL_TEXT"]
    n_tokens = int(os.getenv("VLLM_PPL_TOKENS", "2048"))
    kv_dt = os.getenv("VLLM_PPL_KV_DT", "float16")

    text = open(text_path, encoding="utf-8", errors="ignore").read()
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    ids = tok(text)["input_ids"][: n_tokens + 1]
    if len(ids) < 64:
        raise SystemExit(f"text too short after tokenizing: {len(ids)} tokens")
    # Score a contiguous window; the exact window only matters in that it must
    # be identical across presets.
    prompt_ids = ids[:n_tokens]
    prompt = tok.decode(prompt_ids)
    # Re-tokenize the decoded string so what the engine sees matches the ids we
    # will attribute logprobs to.
    prompt_ids = tok(prompt)["input_ids"]
    n = len(prompt_ids)

    kw = dict(
        model=model,
        tensor_parallel_size=int(os.getenv("VLLM_PPL_TP", "2")),
        gpu_memory_utilization=float(os.getenv("VLLM_PPL_UTIL", "0.98")),
        max_model_len=int(os.getenv("VLLM_PPL_MAXLEN", "8192")),
        max_num_seqs=1,
        max_num_batched_tokens=int(os.getenv("VLLM_PPL_MAX_BT", "256")),
        kv_cache_dtype=kv_dt,
        mamba_ssm_cache_dtype="float16",
        enforce_eager=os.getenv("VLLM_PPL_EAGER", "1") == "1",
        # Weight offload NaN-corrupts the repacked NVFP4 TurboMind weights.
        cpu_offload_gb=0.0,
    )
    if not os.getenv("VLLM_PPL_MAMBA_MODE", ""):
        pass  # do not pass mamba_cache_mode at all (align is not 1Cat parity)
    else:
        kw["mamba_cache_mode"] = os.environ["VLLM_PPL_MAMBA_MODE"]
    if os.getenv("VLLM_PPL_ATTN_BACKEND"):
        kw["attention_backend"] = os.environ["VLLM_PPL_ATTN_BACKEND"]
    if kv_dt.startswith("turboquant"):
        kw.setdefault("attention_backend", "TURBOQUANT")

    llm = LLM(**kw)
    out = llm.generate(
        prompts=[{"prompt_token_ids": prompt_ids}],
        sampling_params=SamplingParams(
            max_tokens=1, temperature=0.0, prompt_logprobs=0
        ),
        use_tqdm=False,
    )
    lps = out[0].prompt_logprobs
    nll: list[float] = []
    for i, step in enumerate(lps):
        if step is None:
            continue
        # prompt_logprobs=0 returns {token_id: Logprob} for the actual token.
        for tok_id, lp in step.items():
            if tok_id is None:
                continue
            if lp.logprob == -math.inf or math.isnan(lp.logprob):
                continue
            nll.append(-lp.logprob)
    if not nll:
        raise SystemExit("no prompt logprobs returned")
    mean_nll = sum(nll) / len(nll)
    print("\n=== SM70 perplexity probe ===")
    print(f"kv_cache_dtype : {kv_dt}")
    print(f"text           : {text_path}")
    print(f"scored tokens  : {len(nll)} / prompt {n}")
    print(f"mean NLL       : {mean_nll:.5f}")
    print(f"perplexity     : {math.exp(mean_nll):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
