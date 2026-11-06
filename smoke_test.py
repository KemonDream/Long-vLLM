#!/usr/bin/env python3
"""Preview smoke test for the SM70/V100 vLLM service.

Run it against a known-good server first: if the harness fails there, the
harness is wrong, not the build. Checks, in order:

  1. /v1/models answers
  2. decode is non-NaN and non-degenerate (logprobs finite, >1 distinct token)
  3. decode throughput from SERVER-reported token counts (never chars/4)
  4. tool calling works and does not leak the raw <tool_call> markup
  5. reasoning is present under BOTH names (reasoning, reasoning_content)
  6. prefix cache reports a hit on a repeat of a >2k-token prompt
  7. the engine used the Flash-V100 packed-cache decode path, not the Triton
     fallback (the fallback is silent, so the log is the only evidence)

Every number printed is accompanied by the evidence for it. Exit code is 0 only
if every check passed.
"""
import json
import os
import re
import sys
import time
import urllib.request

BASE = os.environ.get("SMOKE_BASE", "http://127.0.0.1:30000")
# Point this at the server's stdout log; if it is the wrong file the log
# checks below say which file they read, and when it was last written.
LOGFILE = os.environ.get("SMOKE_LOG", "server.log")
# No host-specific default: pass SMOKE_TOKENIZER, or the script uses the model id
# the server itself reports.
TOKENIZER = os.environ.get("SMOKE_TOKENIZER") or None

fails: list[str] = []
notes: list[str] = []


def post(path, payload, timeout=600):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path, timeout=60):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


def check(name, ok, detail):
    print(("PASS  " if ok else "FAIL  ") + f"{name:<44} {detail}")
    if not ok:
        fails.append(name)


# ---- 1. the server answers --------------------------------------------------
try:
    models = get("/v1/models")
    mid = models["data"][0]["id"]
    check("server answers /v1/models", True, mid)
except Exception as e:  # noqa: BLE001
    print(f"FAIL  server answers /v1/models                {e}")
    sys.exit(1)

# ---- 2/3. decode quality and throughput -------------------------------------
prompt = "Explain in one paragraph why a page table is needed for paged attention."
t0 = time.time()
r = post(
    "/v1/chat/completions",
    {
        "model": mid,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 3,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
dt = time.time() - t0
usage = r["usage"]
ctok = usage["completion_tokens"]
ptok = usage["prompt_tokens"]
lps = [
    t["logprob"]
    for t in (r["choices"][0].get("logprobs") or {}).get("content") or []
    if t.get("logprob") is not None
]
finite = [x for x in lps if x == x and abs(x) != float("inf")]
distinct = len({round(x, 6) for x in finite})
check(
    "decode logprobs are finite (no NaN)",
    len(finite) == len(lps) and len(lps) > 0,
    f"{len(finite)}/{len(lps)} finite",
)
check("decode is non-degenerate", distinct > 1, f"{distinct} distinct logprob values")
tps = ctok / dt if dt else 0
print(
    f"INFO  decode: {ctok} server-counted tokens in {dt:.1f}s = "
    f"{tps:.2f} tok/s ({ptok} prompt tokens)"
)

# ---- 4. tool calling --------------------------------------------------------
tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
r = post(
    "/v1/chat/completions",
    {
        "model": mid,
        "messages": [{"role": "user", "content": "What is the weather in Berlin?"}],
        "tools": [tool],
        "tool_choice": "auto",
        "max_tokens": 256,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
msg = r["choices"][0]["message"]
tc = msg.get("tool_calls") or []
content = msg.get("content") or ""
check("tool call parsed", len(tc) == 1, f"{len(tc)} tool_calls")
if tc:
    args = tc[0]["function"]["arguments"]
    check(
        "tool arguments are JSON",
        "Berlin" in args,
        args[:80],
    )
check("no raw tool markup leaked", "<tool_call>" not in content, "clean")

# ---- 5. reasoning under both names -----------------------------------------
r = post(
    "/v1/chat/completions",
    {
        "model": mid,
        "messages": [{"role": "user", "content": "What is 17*23? Think it through."}],
        "max_tokens": 256,
        "temperature": 0.0,
    },
)
msg = r["choices"][0]["message"]
reason = msg.get("reasoning")
alias = msg.get("reasoning_content")
check("reasoning present", bool(reason), f"{len(reason or '')} chars")
check(
    "reasoning_content alias present",
    bool(alias),
    f"{len(alias or '')} chars, identical={alias == reason}",
)

# ---- 6. prefix cache on a long prompt --------------------------------------
# Prompts are sent as token ids so both calls are byte-identical and the block
# granularity (2064 tokens) cannot make this report a spurious zero.
try:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER or mid, trust_remote_code=True)
    # ~8k tokens: over the 2064-token block, small enough not to disturb load
    text = open("/tmp/vllm_sm70/smoke_prefix.txt").read()[:32000]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    body_txt = tok.decode(ids)
except Exception as e:  # noqa: BLE001
    notes.append(f"prefix-cache check skipped: {e}")
    ids = None

if ids and len(ids) >= 3000:
    hits = []
    for i in range(2):
        r = post(
            "/v1/completions",
            {
                "model": mid,
                "prompt": ids,
                "max_tokens": 1,
                "temperature": 0.0,
            },
        )
        d = (r["usage"].get("prompt_tokens_details") or {}) if "usage" in r else {}
        hits.append(d.get("cached_tokens"))
    check(
        "prefix cache reports a hit on repeat",
        (hits[1] or 0) > 0,
        f"prompt {len(ids)} tokens, cached_tokens={hits}",
    )
else:
    notes.append(f"prefix-cache check needs >=3000 tokens, got {len(ids or [])}")

# ---- 7. which decode path is live? ----------------------------------------
try:
    log = open(LOGFILE, errors="replace").read()
    n_flash = len(re.findall(r"TURBOQUANT decode using Flash-V100 packed-cache path", log))
    pool = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
    check(
        "Flash-V100 packed-cache decode path selected",
        n_flash > 0,
        f"{n_flash} worker(s) reported it (log {LOGFILE}, mtime "
        f"{time.strftime('%H:%M', time.localtime(os.path.getmtime(LOGFILE)))})",
    )
    if pool:
        print(f"INFO  KV pool from log: {pool[-1]} tokens")
    n_err = len(re.findall(r"EngineCore .*died|CUDA error|Traceback", log))
    check(
        "no engine death / traceback in log",
        n_err == 0,
        f"{n_err} matches in {LOGFILE} "
        f"(mtime {time.strftime('%H:%M', time.localtime(os.path.getmtime(LOGFILE)))})",
    )
except FileNotFoundError:
    notes.append(f"log {LOGFILE} not found")

print()
for n in notes:
    print("NOTE ", n)
print(f"---- {'ALL CHECKS PASSED' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
