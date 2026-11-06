#!/usr/bin/env bash
# Checks 1/2/4: CPU KV offload losslessness, whether it actually works, and
# quality under pool-saturating mixed-context concurrency.
#
# Two server configurations, same test battery:
#   A. locked profile, no offload tier      -> reference + nondeterminism floor
#   B. same profile + --kv-offloading-size  -> stress with heavy eviction
set -u
TREE=${SM70_TREE:?set SM70_TREE to the directory containing venv/ and vllm/}
OUT=/tmp/vllm_sm70/oq
mkdir -p "$OUT"
PY="$TREE/venv/bin/python"
export VLLM_OQ_MAXTOK=48

clean() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$p" 2>/dev/null
  done
  sleep 12
}

wait_health() {  # port
  local i code
  for i in $(seq 1 100); do
    code=$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$1/health" || true)
    [ "$code" = "200" ] && { echo "  health 200"; return 0; }
    sleep 5
  done
  echo "  health FAILED"; return 1
}

run_server_a() {
  clean
  VLLM_SM70_PORT=30000 nohup ${SM70_LAUNCH_A:?set SM70_LAUNCH_A to the TurboQuant profile launcher} \
      > "$OUT/server_a.log" 2>&1 &
  wait_health 30000
}

run_server_b() {
  clean
  VLLM_SM70_PORT=30001 OFFLOAD_GB=${OFFLOAD_GB:-24} \
      nohup ${SM70_LAUNCH_B:?set SM70_LAUNCH_B to the CPU-KV-offload profile launcher} > "$OUT/server_b.log" 2>&1 &
  wait_health 30001
}

echo "===== A: locked profile, offload OFF ====="
run_server_a
for i in 1 2; do
  echo "  record pass $i"
  VLLM_OQ_BASE=http://127.0.0.1:30000 VLLM_OQ_LENS=4096,16384,8192,32768 \
    "$PY" "$(dirname "$0")/sm70_offload_quality.py" record --out "$OUT/rec_$i.json" \
    2>&1 | grep -E '^  L|corpus'
done
grep -E 'GPU KV cache size' "$OUT/server_a.log" | tail -1

echo "===== B: same profile + CPU KV offload ====="
run_server_b
grep -E 'GPU KV cache size|offload|Offload|CPU KV' "$OUT/server_b.log" | tail -4
VLLM_OQ_BASE=http://127.0.0.1:30001 VLLM_OQ_LENS=4096,16384,8192,32768 \
  "$PY" "$(dirname "$0")/sm70_offload_quality.py" stress --out "$OUT/stress.json" \
  2>&1 | grep -vE '^\s*$'

echo "===== compare ====="
"$PY" "$(dirname "$0")/sm70_offload_quality.py" compare \
  "$OUT/rec_1.json" "$OUT/stress.json" 2>&1 | tail -40
echo "===== nondeterminism floor (A pass1 vs A pass2) ====="
"$PY" "$(dirname "$0")/sm70_offload_quality.py" compare \
  "$OUT/rec_1.json" "$OUT/rec_2.json" 2>&1 | sed -n '/sequential/,/^$/p'
echo OFFLOAD_DONE
