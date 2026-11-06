#!/usr/bin/env python
"""CPU-KV-offload quality harness: is the offload path lossless, and does a hot
restore actually buy prefill time?

Two runs against two server configurations, then a paired comparison:

    # server A: no CPU KV offload
    python harnesses/sm70_offload_quality.py record --out /tmp/vllm_sm70/oq_off.json
    # server B: --kv-offloading-size N, heavy eviction
    python harnesses/sm70_offload_quality.py stress --out /tmp/vllm_sm70/oq_on.json
    python harnesses/sm70_offload_quality.py compare /tmp/vllm_sm70/oq_off.json \
                                                  /tmp/vllm_sm70/oq_on.json

`record` replays each stream alone (baseline). `stress` replays them one at a time
too (so the offload path is exercised without eviction), then re-runs them all
concurrently plus filler traffic sized to overrun the GPU pool, forcing blocks out
to CPU and back mid-generation.

Determinism note: greedy decoding is not bit-reproducible across different batch
compositions, so `compare` reports the divergence, and `record` can be run twice to
establish the no-offload nondeterminism floor. Offload is called lossless only if
the offload divergence is no worse than that floor.

Everything is measured from server-reported `usage` counts, never from characters.
"""

import argparse
import glob
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

MODEL = os.getenv("VLLM_OQ_MODEL", "qwen38-27b-nvfp4")
CORPUS = os.getenv("VLLM_OQ_CORPUS", "/tmp/vllm_sm70/quality_corpus.txt")
BASE = os.getenv("VLLM_OQ_BASE", "http://127.0.0.1:30000")
GEN = int(os.getenv("VLLM_OQ_MAXTOK", "64"))

# Mixed context lengths: short/mid/long interleaved so the pool sees heterogeneous
# blocks, which is what a real mixed workload looks like.
LENS = [int(x) for x in os.getenv("VLLM_OQ_LENS", "4096,16384,8192,32768").split(",")]


def token_ids():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        os.getenv("VLLM_OQ_TOKENIZER",
                  "/path/to/model"),
        trust_remote_code=False,
    )
    text = open(CORPUS, encoding="utf-8", errors="ignore").read()
    return tok(text)["input_ids"]


def post(path, payload, timeout=7200):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def metrics():
    """Scrape /metrics and return prompt-token counters by source, plus KV state."""
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=30) as h:
            text = h.read().decode("utf-8", "ignore")
    except Exception:
        return {}
    out = {"source": {}, "raw": {}}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("vllm:prompt_tokens_by_source_total"):
            src = None
            if 'source="' in line:
                src = line.split('source="', 1)[1].split('"', 1)[0]
            try:
                out["source"][src] = out["source"].get(src, 0.0) + float(
                    line.rsplit(" ", 1)[1]
                )
            except (ValueError, IndexError):
                pass
        elif line.startswith(("vllm:kv_cache_usage_perc",
                              "vllm:gpu_cache_usage_perc",
                              "vllm:num_requests_running",
                              "vllm:num_requests_waiting",
                              "vllm:prefix_cache_hits_total",
                              "vllm:prefix_cache_queries_total",
                              "vllm:external_prefix_cache_hits_total",
                              "vllm:prompt_tokens_cached_total")):
            name = line.split("{", 1)[0].split(" ", 1)[0]
            try:
                v = float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                continue
            out["raw"][name] = out["raw"].get(name, 0.0) + v
    return out


def gen(prompt_ids, stream=False, max_tokens=None):
    """One completion. Returns ids, logprobs, usage, and ttft."""
    payload = {
        "model": MODEL,
        "prompt": list(prompt_ids),
        "max_tokens": GEN if max_tokens is None else max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "top_p": 1.0,
        "logprobs": 1,
        "stream": stream,
    }
    t0 = time.perf_counter()
    if not stream:
        with post("/v1/completions", payload) as h:
            d = json.load(h)
        dt = time.perf_counter() - t0
        ch = d["choices"][0]
        lp = ch.get("logprobs") or {}
        return {
            "ids": list(lp.get("tokens") or []),
            "tokens": lp.get("tokens") or [],
            "logprobs": [v for v in (lp.get("token_logprobs") or [])
                         if v is not None],
            "usage": d["usage"],
            "wall_s": dt,
            "ttft_s": None,
            "text": ch.get("text", ""),
        }
    ttft = None
    ids, lps, usage, text = [], [], None, []
    with post("/v1/completions", payload) as h:
        for raw in h:
            if not raw.startswith(b"data: "):
                continue
            body = raw[6:].strip()
            if body == b"[DONE]":
                break
            d = json.loads(body)
            if usage is None and "usage" in d and d["usage"]:
                usage = d["usage"]
            for ch in d.get("choices") or []:
                piece = ch.get("text")
                if piece and ttft is None:
                    ttft = time.perf_counter() - t0
                if piece:
                    text.append(piece)
                lp = ch.get("logprobs") or {}
                lps += [v for v in (lp.get("token_logprobs") or [])
                        if v is not None]
                ids += list(lp.get("tokens") or [])
    return {
        "ids": ids,
        "tokens": [],
        "logprobs": lps,
        "usage": usage,
        "wall_s": time.perf_counter() - t0,
        "ttft_s": ttft,
        "text": "".join(text),
    }


def longest_repeat(tokens):
    """Longest run of one repeated token: a cheap degenerate-loop detector."""
    best = run = 1
    for i in range(1, len(tokens)):
        run = run + 1 if tokens[i] == tokens[i - 1] else 1
        best = max(best, run)
    return best if tokens else 0


def summarize(name, r):
    n = len(r["logprobs"])
    return {
        "stream": name,
        "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"),
        "completion_tokens": (r["usage"] or {}).get("completion_tokens"),
        "ids": r["ids"],
        "logprobs": r["logprobs"],
        "mean_logprob": (sum(r["logprobs"]) / n) if n else None,
        "distinct_tokens": len(set(r["ids"])),
        "n_tokens": n,
        "longest_repeat": longest_repeat(r["ids"]),
        "wall_s": r["wall_s"],
        "ttft_s": r["ttft_s"],
        "text_head": (r["text"] or "")[:48].replace("\n", " "),
    }


def do_record(ids):
    res = {"mode": "record", "metrics_before": metrics(), "streams": []}
    for i, L in enumerate(LENS):
        if L >= len(ids):
            continue
        r = gen(ids[:L])
        s = summarize(f"L{L}", r)
        res["streams"].append(s)
        print(f"  L{L:6d} prompt={s['prompt_tokens']} "
              f"meanlogp={s['mean_logprob']:+.4f} distinct={s['distinct_tokens']}"
              f"/{s['n_tokens']} rep={s['longest_repeat']} {s['wall_s']:.1f}s")
    res["metrics_after"] = metrics()
    return res


def do_stress(ids, fillers=6):
    res = {"mode": "stress", "metrics_before": metrics(), "streams": [],
            "burst": [], "hot": {}}

    # Phase A: same sequential replay as `record`, so the only difference from the
    # reference is that the offload machinery is installed.
    print("  phase A: sequential replay (offload installed, no eviction)")
    for L in LENS:
        if L >= len(ids):
            continue
        r = gen(ids[:L])
        s = summarize(f"seq_L{L}", r)
        res["streams"].append(s)
        print(f"    L{L:6d} meanlogp={s['mean_logprob']:+.4f} "
              f"distinct={s['distinct_tokens']}/{s['n_tokens']}")

    # Phase B: everything at once, plus filler sized to overrun the GPU pool.
    # Eviction is what forces blocks to CPU; without overrun this proves nothing.
    print(f"  phase B: concurrent burst {len(LENS)} streams + {fillers} fillers")
    payloads = [(f"L{L}", ids[:L]) for L in LENS if L < len(ids)]
    payloads += [(f"fill{j}",
                  ids[(j + 1) * 9000: (j + 1) * 9000 + LENS[-1]])
                 for j in range(fillers)]
    results, errs = {}, []
    sem = threading.Semaphore(int(os.getenv("VLLM_OQ_FLOOD", "4")))

    def worker(tag, pids):
        try:
            with sem:
                results[tag] = gen(pids, stream=False)
        except Exception as exc:  # noqa: BLE001
            errs.append(f"{tag}: {exc}")

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=p) for p in payloads]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # A burst failure must not lose the run: record it and continue to phase C.
    burst_s = time.perf_counter() - t0
    res["metrics_peak"] = metrics()
    for tag, r in results.items():
        if tag.startswith("fill"):
            continue
        s = summarize(tag, r)
        s["burst"] = True
        res["burst"].append(s)
        print(f"    {tag:10s} prompt={s['prompt_tokens']} "
              f"meanlogp={(s['mean_logprob'] if s['mean_logprob'] is not None else float('nan')):+.4f} "
              f"distinct={s['distinct_tokens']}/{s['n_tokens']} "
              f"rep={s['longest_repeat']} ttft={s['ttft_s']}")
    res["burst_wall_s"] = burst_s
    res["errors"] = errs

    # Phase C: is a restored-from-CPU prefix actually faster than prefilling it?
    L = LENS[-1]
    if L < len(ids):
        print("  phase C: hot-vs-cold TTFT for a restored prefix")
        flood = [("f", ids[j * 1000: j * 1000 + L]) for j in range(fillers)]
        def _flood(p):
            try:
                with sem:
                    gen(p[1], max_tokens=1)
            except Exception as exc:  # noqa: BLE001
                errs.append(f"flood {p[0]}: {exc}")

        th = [threading.Thread(target=_flood, args=(p,)) for p in flood]
        for t in th:
            t.start()
        for t in th:
            t.join()
        try:
            cold = gen(ids[L:L + L], stream=True, max_tokens=1)
        except Exception as exc:  # noqa: BLE001
            errs.append(f"cold probe: {exc}")
            cold = None
        rcold = summarize(f"hot_cold_L{L}", cold) if cold else {"ttft_s": None}
        res["hot"]["cold_ttft_s"] = rcold["ttft_s"]
        try:
            hot = gen(ids[:L], stream=True, max_tokens=1)
        except Exception as exc:  # noqa: BLE001
            errs.append(f"hot probe: {exc}")
            hot = None
        rhot = summarize(f"hot_warm_L{L}", hot) if hot else {"ttft_s": None, "prompt_tokens": None}
        res["hot"]["hot_ttft_s"] = rhot["ttft_s"]
        res["hot"]["prompt_tokens"] = rhot["prompt_tokens"]
        print(f"    cold ttft {rcold['ttft_s']}  hot ttft {rhot['ttft_s']}")
    res["metrics_after"] = metrics()
    return res


def do_compare(pa, pb):
    A, B = json.load(open(pa)), json.load(open(pb))

    def by_name(d, field):
        return {s["stream"]: s for s in d.get(field, [])}

    print(f"A = {pa}  ({A.get('mode')})")
    print(f"B = {pb}  ({B.get('mode')})\n")

    def diff(a, b, label):
        if a is None or b is None:
            return
        ia, ib = a["ids"], b["ids"]
        n = min(len(ia), len(ib))
        same = sum(1 for i in range(n) if ia[i] == ib[i])
        la, lb = a["logprobs"], b["logprobs"]
        m = min(len(la), len(lb))
        if m:
            ds = [abs(la[i] - lb[i]) for i in range(m)]
            md = sum(ds) / m
            mx = max(ds)
        else:
            md = mx = float("nan")
        print(f"  {label:26s} token_match {same}/{n} ({same/max(n,1)*100:.1f}%)  "
              f"mean|dlogp| {md:.3e}  max|dlogp| {mx:.3e}  "
              f"rep {a['longest_repeat']}/{b['longest_repeat']}")

    print("=== sequential streams: offload-installed vs reference ===")
    ref = by_name(A, "streams")
    for s in B.get("streams", []):
        name = s["stream"].replace("seq_", "")
        diff(ref.get(name), s, name)

    print("\n=== concurrent burst under eviction vs reference ===")
    for s in B.get("burst", []):
        diff(ref.get(s["stream"]), s, s["stream"])

    print("\n=== prompt token sources (what the offload actually served) ===")
    for tag, d in (("A", A), ("B", B)):
        ends = [d.get("metrics_after"), d.get("metrics_peak")]
        print(f"  {tag}: " + "  ".join(
            f"{k}={v:,.0f}" for k, v in sorted(
                (ends[1] or ends[0] or {}).get("source", {}).items())))
    hot = B.get("hot") or {}
    if hot.get("cold_ttft_s") and hot.get("hot_ttft_s"):
        c, h = hot["cold_ttft_s"], hot["hot_ttft_s"]
        print(f"\n=== hot restore vs cold prefill ({hot.get('prompt_tokens')} tok) ===")
        pt = hot.get("prompt_tokens")
        print(f"  cold {c:.2f}s -> hot {h:.2f}s  speedup {c/h:.2f}x")
        if pt:
            print(f"  => hot path {pt/h:,.0f} tok/s effective")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "stress", "compare"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--out", default="")
    ap.add_argument("--fillers", type=int, default=6)
    a = ap.parse_args()

    if a.mode == "compare":
        if len(a.paths) != 2:
            raise SystemExit("compare needs two json paths")
        return do_compare(a.paths[0], a.paths[1])

    ids = token_ids()
    print(f"corpus tokens: {len(ids):,}  lens: {LENS}  base: {BASE}")
    res = do_record(ids) if a.mode == "record" else do_stress(ids, a.fillers)
    res["lens"] = LENS
    res["gen_max_tokens"] = GEN
    out = a.out or f"/tmp/vllm_sm70/oq_{a.mode}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
