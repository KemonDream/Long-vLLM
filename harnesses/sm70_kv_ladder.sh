#!/usr/bin/env bash
# KV-quantisation quality ladder on SM70 (V100), TP2.
#
# The point of the ladder is to separate the two quantiser knobs:
#   keys:   fp16 -> fp8(k8v4) -> Lloyd-Max 4-bit (4bit_nc) -> Lloyd-Max 3-bit (k3v4_nc)
#   values: 4-bit (k8v4/4bit_nc/k3v4_nc) -> 3-bit (3bit_nc)
# so that  k3v4_nc - 4bit_nc  isolates the KEY bit, and
#         3bit_nc - k3v4_nc  isolates the VALUE bit.
#
# max_num_batched_tokens is pinned low on purpose: prompt_logprobs materialises
# logits for the whole chunk (chunk x 248k vocab), and at 2048 that alone OOMs.
set -u
TREE=${SM70_TREE:?set SM70_TREE to the directory containing venv/ and vllm/}
OUT=/tmp/vllm_sm70/nll
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NLL_MAX_BT=256
export VLLM_NLL_UTIL=0.90

run() {  # kv attn window
  local kv=$1 attn=$2 win=$3
  local tag="${kv}_w${win}"
  local log="$OUT/$tag.log"
  echo "##### $tag #####"
  # Clean slate: orphaned workers are what silently poison these runs.
  pkill -f 'sm70_kv_nll' 2>/dev/null; sleep 3
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$p" 2>/dev/null
  done
  sleep 8
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')
  echo "  gpu_before: $used"
  VLLM_NLL_KV=$kv VLLM_NLL_ATTN=$attn VLLM_NLL_WINDOW=$win \
  VLLM_NLL_MAXLEN=$((win + 16)) \
  VLLM_NLL_OUT="$OUT/$tag.json" VLLM_NLL_NPY="$OUT/$tag.npy" \
  timeout 3600 "$TREE/venv/bin/python" "$(dirname "$0")/sm70_kv_nll.py" \
      > "$log" 2>&1
  local rc=$?
  if [ -f "$OUT/$tag.json" ]; then
    "$TREE/venv/bin/python" - "$OUT/$tag.json" "$log" <<'PY'
import json, re, sys
d = json.load(open(sys.argv[1]))
log = open(sys.argv[2], errors="ignore").read()
pool = re.search(r"GPU KV cache size: ([\d,]+) tokens", log)
sel = re.findall(r"Using (.*?) GDN prefill kernel", log)
print(f"  mean_nll {d['mean_nll']:.6f}  se {d['se']:.6f}  ppl {d['ppl']:.4f}")
print(f"  scored {d['scored_tokens']}  prompt {d['prompt_tokens']}  "
      f"pool {pool.group(1) if pool else '?'}  distinct {d['distinct_scored']}  "
      f"finite {d['finite']}")
print(f"  prefill {d['prefill_tok_s']:.0f} tok/s  wall {d['wall_s']:.1f}s  "
      f"gdn {sel[0] if sel else '?'}")
PY
  else
    echo "  FAILED rc=$rc"; grep -iE 'OutOfMemory|RuntimeError|Error:' "$log" | tail -3
  fi
}

# fp16 KV needs a non-TurboQuant backend; it is the only absolute reference we
# have, but it cannot reach long context (its pool is ~4x smaller).
run float16              FLASH_ATTN_V100 8192
for kv in turboquant_k8v4 turboquant_4bit_nc turboquant_k3v4_nc turboquant_3bit_nc; do
  run "$kv" TURBOQUANT 8192
done
for kv in turboquant_k8v4 turboquant_4bit_nc turboquant_k3v4_nc turboquant_3bit_nc; do
  run "$kv" TURBOQUANT 32768
done
for kv in turboquant_k3v4_nc turboquant_3bit_nc; do
  run "$kv" TURBOQUANT 65536
done
run turboquant_3bit_nc TURBOQUANT 131072
echo LADDER_DONE
