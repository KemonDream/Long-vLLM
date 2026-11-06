#!/usr/bin/env python
"""Teacher-forced logprob dump for A/B-ing GDN prefill backends.

Feeds the SAME token sequence and records the model's own per-token logprob, so the
comparison is free of sampling/tie-breaking noise. Two runs (env toggled) then diff:

    VLLM_SM70_GDN_FLASHQLA_PREFILL=1 python harnesses/sm70_flashqla_logprob.py /tmp/a.json
    VLLM_SM70_GDN_FLASHQLA_PREFILL=0 python harnesses/sm70_flashqla_logprob.py /tmp/b.json
"""

import json
import os
import sys

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "vllm"))

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("VLLM_FQ_MODEL", "/path/to/model")

TEXT = (
    "The history of computing is a history of abstraction. Each layer hides the "
    "details of the layer beneath it, and each generation of engineers builds on "
    "the previous one without needing to understand it fully. Quantization is one "
    "such abstraction: it replaces a wide number with a narrow one, and trusts that "
    "the downstream computation will not notice. Whether that trust is justified "
    "depends on the data, the scale, and the error budget of the task at hand. "
    "In practice the errors are small, correlated, and mostly harmless, but the "
    "argument for that claim is empirical rather than mathematical.\n\n"
    "A second theme is parallelism. Splitting a matrix across devices trades "
    "communication for compute, and the exchange rate between the two depends "
    "entirely on the interconnect. When the link is fast the split is nearly free; "
    "when the link is slow the split can dominate the whole calculation. The same "
    "code, run on two machines with different fabrics, can differ by a factor of "
    "several, and no amount of kernel tuning closes that gap.\n\n"
    "A third theme is measurement. A number without a denominator is an anecdote."
)


def main(out_path):
    llm = LLM(
        model=MODEL,
        dtype="float16",
        tensor_parallel_size=2,
        kv_cache_dtype=os.getenv("VLLM_SM70_KV", "turboquant_3bit_nc"),
        mamba_ssm_cache_dtype="float16",
        attention_backend="TURBOQUANT",
        max_model_len=16384,
        max_num_seqs=2,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.98,
        enable_prefix_caching=False,
        trust_remote_code=True,
        compilation_config={
            "mode": "NONE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2],
        },
    )

    tok = llm.get_tokenizer()
    n = len(tok.encode(TEXT))

    # prompt_logprobs: the model scores the given tokens itself (no sampling).
    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0,
                        ignore_eos=True)
    r = llm.generate([TEXT], sp)[0]

    per_token = []
    for i, step in enumerate(r.prompt_logprobs or []):
        if not step:
            per_token.append(None)
            continue
        vals = list(step.values())
        per_token.append(vals[0].logprob if vals else None)

    kept = [v for v in per_token if v is not None]
    out = {
        "flashqla": os.getenv("VLLM_SM70_GDN_FLASHQLA_PREFILL", "1"),
        "prompt_tokens": n,
        "scored": len(kept),
        "any_nan": any(v != v for v in kept),
        "mean_logprob": sum(kept) / len(kept) if kept else None,
        # keep it compact: every 4th scored value is plenty to detect drift
        "logprobs": kept[::1],
    }
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"WROTE {out_path} flashqla={out['flashqla']} tokens={n} "
          f"scored={len(kept)} mean={out['mean_logprob']:.6f} nan={out['any_nan']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/lp.json"))
