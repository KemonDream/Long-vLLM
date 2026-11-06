"""Aggregate a vLLM torch-profiler trace into per-kernel CUDA time.

Traces land as `rankN.<id>.pt.trace.json.gz`; this reads them directly so a
profile run does not have to be repeated just to parse it.
"""

import glob
import gzip
import json
import os
import sys


def load(path: str):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return json.load(fh)
    with open(path, encoding="utf-8", errors="replace") as fh:
        return json.load(fh)


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vllm_sm70/prof"
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    traces = sorted(glob.glob(os.path.join(root, "**", "*.json*"), recursive=True))
    if not traces:
        raise SystemExit(f"no traces under {root}")
    agg: dict[str, list[float]] = {}
    for path in traces:
        try:
            data = load(path)
        except Exception as exc:
            print(f"skip {path}: {exc}")
            continue
        events = data.get("traceEvents", data if isinstance(data, list) else [])
        for ev in events:
            if ev.get("cat") not in ("kernel", "Kernel", "gpu_user_annotation"):
                continue
            dur = float(ev.get("dur", 0.0) or 0.0)
            if dur <= 0:
                continue
            agg.setdefault(ev.get("name", "?"), []).append(dur)
    if not agg:
        raise SystemExit("no kernel events found")
    total = sum(sum(v) for v in agg.values())
    rows = sorted(agg.items(), key=lambda kv: -sum(kv[1]))
    print(f"traces: {len(traces)}  kernels: {len(agg)}  total CUDA: {total/1000:.1f} ms\n")
    print(f"{'kernel':64s} {'ms':>9s} {'calls':>7s} {'%':>6s}")
    for name, durs in rows[:top]:
        print(
            f"{name[:64]:64s} {sum(durs)/1000:9.1f} {len(durs):7d} "
            f"{100*sum(durs)/total:6.1f}"
        )
    # Group by family so the "should we port GDN?" question is answerable.
    fams = {
        "gdn/fla": ("gdn", "delta", "fla", "conv1d", "causal_conv", "l2norm"),
        "nvfp4-gemv": ("nvfp4", "qpn", "turbomind", "gemv", "cutlass"),
        "attention": ("attention", "flash", "turboquant", "xqa", "paged"),
        "norm/act": ("rms", "silu", "gelu", "norm"),
        "gemm-other": ("gemm", "matmul", "mm_", "swiglu", "linear"),
        "nccl": ("nccl", "allreduce", "all_reduce"),
    }
    print("\n--- families ---")
    for fam, pats in fams.items():
        tot = sum(
            sum(v) for k, v in agg.items() if any(p in k.lower() for p in pats)
        )
        print(f"{fam:12s} {tot/1000:9.1f} ms {100*tot/total:6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
