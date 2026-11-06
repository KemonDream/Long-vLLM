#!/usr/bin/env python3
"""Decode benchmark for the SM70 port: pool size + tok/s at a given KV dtype.

Mirrors the former 1Cat launcher knobs (see the 1Cat-vLLM launcher) so
numbers are comparable with the ones recorded during development:

Usage (both cards, production parity, fp8 baseline):
  VLLM_BM_MODEL=/path/to/model \
  VLLM_BM_TP=2 VLLM_BM_UTIL=0.97 VLLM_BM_MAXLEN=163840 \
  VLLM_BM_KV_DT=fp8_e4m3 VLLM_BM_TOKENS=128 \
  ./venv/bin/python harnesses/sm70_bench.py

The engine log is scanned for the KV pool line so capacity and speed land in
one report instead of two.
"""

import os
import time


def main() -> int:
    from vllm import LLM, SamplingParams

    model = os.environ["VLLM_BM_MODEL"]
    kv_dt = os.getenv("VLLM_BM_KV_DT", "fp8_e4m3")
    kw = dict(
        model=model,
        tensor_parallel_size=int(os.getenv("VLLM_BM_TP", "2")),
        pipeline_parallel_size=int(os.getenv("VLLM_BM_PP", "1")),
        gpu_memory_utilization=float(os.getenv("VLLM_BM_UTIL", "0.97")),
        max_model_len=int(os.getenv("VLLM_BM_MAXLEN", "163840")),
        max_num_seqs=int(os.getenv("VLLM_BM_MAXSEQS", "16")),
        enforce_eager=os.getenv("VLLM_BM_EAGER", "0") == "1",
        kv_cache_dtype=kv_dt,
        mamba_cache_mode=os.getenv("VLLM_BM_MAMBA_MODE", "align"),
        mamba_ssm_cache_dtype=os.getenv("VLLM_BM_MAMBA_SSM_DT", "float16"),
        cpu_offload_gb=float(os.getenv("VLLM_BM_CPU_OFFLOAD_GB", "0.9")),
        trust_remote_code=os.getenv("VLLM_BM_TRUST_REMOTE_CODE", "0") == "1",
    )
    # An empty VLLM_BM_MAMBA_MODE means "do not pass the flag at all", which is
    # what the 1Cat production launcher did (it never set mamba_cache_mode).
    if not os.getenv("VLLM_BM_MAMBA_MODE", "align"):
        kw.pop("mamba_cache_mode", None)

    attn = os.getenv("VLLM_BM_ATTN_BACKEND")
    if attn:
        kw["attention_backend"] = attn
    if os.getenv("VLLM_BM_MAX_BT"):
        kw["max_num_batched_tokens"] = int(os.environ["VLLM_BM_MAX_BT"])
    if os.getenv("VLLM_BM_LINEAR_BACKEND"):
        kw["linear_backend"] = os.environ["VLLM_BM_LINEAR_BACKEND"]

    if os.getenv("VLLM_BM_NANWATCH", "0") == "1":
        # Find the first submodule whose forward output contains NaN. Requires
        # VLLM_ENABLE_V1_MULTIPROCESSING=0 so the model lives in this process.
        import torch as _t

        _seen: list[str] = []

        _skip = tuple(
            x for x in os.getenv("VLLM_BM_NANWATCH_SKIP", "").split(",") if x
        )

        def _hook(mod, _inp, out):
            name = type(mod).__name__
            if len(_seen) >= 40 or name in _seen:
                return
            if any(s in name for s in _skip):
                return
            tensors = out if isinstance(out, (tuple, list)) else (out,)
            for t in tensors:
                if isinstance(t, _t.Tensor) and t.is_floating_point():
                    if not _t.isfinite(t).all():
                        _seen.append(name)
                        print(
                            f"[nanwatch] NaN at {name} "
                            f"prefix={getattr(mod, 'prefix', '')!r} "
                            f"shape={tuple(t.shape)}",
                            flush=True,
                        )
                        break

        _t.nn.modules.module.register_module_forward_hook(_hook)

    if os.getenv("VLLM_BM_PROFILE", "0") == "1":
        # This tree refuses start_profile() unless the engine was built with a
        # profiler config; VLLM_TORCH_PROFILER_DIR alone is not enough.
        from vllm.config.profiler import ProfilerConfig

        kw["profiler_config"] = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=os.environ.get(
                "VLLM_TORCH_PROFILER_DIR", "/tmp/vllm_sm70/prof"
            ),
        )

    if os.getenv("VLLM_BM_NO_COMPILE_GRAPH"):
        # 1Cat's cheapest decode-graph lane: no torch.compile at all, just
        # FULL_DECODE_ONLY capture of the eager decode step. The compile path
        # needs ~2GB this box does not have; replaying eager kernels still
        # removes hundreds of per-step launches.
        from vllm.config.compilation import CompilationConfig, CUDAGraphMode

        kw["compilation_config"] = CompilationConfig(
            mode="NONE",
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[
                int(v) for v in os.getenv("VLLM_BM_CG_SIZES", "1,2").split(",")
            ],
        )
    elif os.getenv("VLLM_BM_CG_MODE"):
        # Full cudagraph capture needs ~2GB the 27B does not have on 2x16GB
        # (measured -1.94 GiB), but PIECEWISE only captures the attention blocks
        # and is far cheaper, so it is worth a separate knob.
        from vllm.config.compilation import CompilationConfig, CUDAGraphMode

        kw["compilation_config"] = CompilationConfig(
            cudagraph_mode=CUDAGraphMode[os.environ["VLLM_BM_CG_MODE"]],
            cudagraph_capture_sizes=[
                int(v) for v in os.getenv("VLLM_BM_CG_SIZES", "1,2,4,8,16").split(",")
            ],
        )

    if os.getenv("VLLM_BM_NO_PREFIX_CACHE", "0") == "1":
        # A warmup request on the same prompt seeds the prefix cache, after which
        # a "prefill" measurement is really a cache hit.
        kw["enable_prefix_caching"] = False

    llm = LLM(**kw)
    cfg = llm.llm_engine.vllm_config
    pool = getattr(getattr(cfg, "cache_config", None), "num_gpu_blocks", None)

    if os.getenv("VLLM_BM_POOL_ONLY", "0") == "1":
        # Capacity-only run: report the pool and stop before generating, so a
        # sweep over kv-cache presets does not pay decode overhead twice.
        print("\n=== SM70 pool only ===")
        print(f"kv_cache_dtype   : {kv_dt}")
        print(f"attention_backend: {kw.get('attention_backend', 'auto')}")
        print(f"max_model_len    : {kw['max_model_len']}")
        print(f"max_num_seqs     : {kw['max_num_seqs']}")
        print(f"gpu blocks       : {pool}")
        return 0

    prompt = os.getenv("VLLM_BM_PROMPT", "Explain quantization in one sentence.")
    prompt = prompt * int(os.getenv("VLLM_BM_PROMPT_REPEAT", "1"))
    conc = int(os.getenv("VLLM_BM_CONCURRENCY", "1"))
    n_tokens = int(os.getenv("VLLM_BM_TOKENS", "128"))
    temp = float(os.getenv("VLLM_BM_TEMP", "0.0"))
    params = SamplingParams(
        temperature=temp,
        max_tokens=n_tokens,
        ignore_eos=True,
        logprobs=int(os.getenv("VLLM_BM_LOGPROBS", "0")) or None,
    )

    prompts = [prompt] * conc

    # Warmup (graph capture / kernel autotune) is excluded from the timing.
    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8))

    if os.getenv("VLLM_BM_PROFILE", "0") == "1":
        # Attribute decode time to kernels before deciding what to port: a
        # 7k-line GDN port is only worth it if the GDN kernels dominate.
        # TP>1 means the work happens in worker processes, so use the engine's
        # own torch-profiler hooks (VLLM_TORCH_PROFILER_DIR) instead of a local
        # profiler context, which would only see the parent process.
        import glob
        import json

        trace_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR", "/tmp/vllm_sm70/prof")
        os.makedirs(trace_dir, exist_ok=True)
        # Profile the actual concurrency: a single-request profile would report
        # bs=1 costs while the measurement is about bs=N throughput.
        # VLLM_BM_PROFILE_DECODE=0 profiles a single long-prompt prefill instead,
        # which is what the prefill question needs.
        ptoks = 1 if os.getenv("VLLM_BM_PROFILE_DECODE", "1") == "1" else 1
        llm.start_profile()
        llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=ptoks,
                                            ignore_eos=True))
        llm.stop_profile()
        traces = sorted(glob.glob(os.path.join(trace_dir, "**", "*.json*"),
                                  recursive=True),
                        key=os.path.getmtime)
        print("\n=== SM70 decode profile ===")
        print(f"traces: {len(traces)}")
        agg: dict[str, list[float]] = {}
        for path in traces[-2:]:
            try:
                with open(path) as fh:
                    data = json.load(fh)
            except Exception as exc:  # gzipped or partial
                print(f"skip {path}: {exc}")
                continue
            events = data.get("traceEvents", data if isinstance(data, list) else [])
            for ev in events:
                if ev.get("cat") not in ("kernel", "Kernel", "gpu_user_annotation"):
                    continue
                name = ev.get("name", "?")
                dur = float(ev.get("dur", 0.0))
                agg.setdefault(name, []).append(dur)
        rows = sorted(agg.items(), key=lambda kv: -sum(kv[1]))
        total = sum(sum(v) for v in agg.values()) or 1.0
        print(f"{'kernel':60s} {'ms':>9s} {'calls':>7s} {'%':>6s}")
        for name, durs in rows[:25]:
            print(f"{name[:60]:60s} {sum(durs)/1000:9.1f} {len(durs):7d} "
                  f"{100*sum(durs)/total:6.1f}")

    t0 = time.perf_counter()
    out = llm.generate(prompts, params)
    dt = time.perf_counter() - t0
    total_produced = sum(len(o.outputs[0].token_ids) for o in out)
    text = out[0].outputs[0].text
    produced = len(out[0].outputs[0].token_ids)
    prompt_tokens = len(out[0].prompt_token_ids)

    print("\n=== SM70 decode bench ===")
    print(f"model            : {model}")
    print(f"kv_cache_dtype   : {kv_dt}")
    print(f"attention_backend: {kw.get('attention_backend', 'auto')}")
    print(f"max_model_len    : {kw['max_model_len']}")
    print(f"util / eager     : {kw['gpu_memory_utilization']} / {kw['enforce_eager']}")
    print(f"gpu blocks       : {pool}")
    print(f"prompt tokens    : {prompt_tokens}")
    print(f"decode tokens    : {produced} in {dt:.2f}s -> {produced / dt:.2f} tok/s")
    if conc > 1:
        # Aggregate vs per-stream: batching efficiency is the whole question on
        # a weight-read-bound box, so report both rather than one number.
        print(f"concurrency      : {conc} requests")
        print(f"aggregate        : {conc} x {produced} = {total_produced} tokens "
              f"in {dt:.2f}s -> {total_produced / dt:.2f} tok/s")
        print(f"per-stream       : {total_produced / dt / conc:.2f} tok/s")
    print(f"temperature      : {temp}")
    print(f"first 12 ids     : {out[0].outputs[0].token_ids[:12]}")
    print(f"distinct ids     : {len(set(out[0].outputs[0].token_ids))}/{produced}")
    lps = getattr(out[0].outputs[0], "logprobs", None) or []
    if lps:
        for i, step in enumerate(lps[:2]):
            if step is None:
                continue
            top = sorted(step.items(), key=lambda kv: -(kv[1].logprob))[:5]
            print(
                f"step {i} top5     : "
                + ", ".join(f"{k}={v.logprob:.3f}" for k, v in top)
            )
    print(f"first 160 chars  : {text[:160]!r}")
    sanity = len(text.strip()) > 0 and text.count(text.strip()[:12]) <= 1
    print(f"sane (non-empty, not looping): {sanity}")
    return 0 if sanity else 1


if __name__ == "__main__":
    raise SystemExit(main())
