# Long-vLLM

**V100 (SM70) support for current upstream vLLM - with a 3-bit quantized KV cache
that fits a 262,144-token context on two 16 GB cards.**

Upstream vLLM dropped Volta. A pair of V100-SXM2-16GB cards is still cheap, fast
silicon for a 27B model - *if the KV cache fits*. That is what this patch set
delivers: SM70 kernel work ported onto **upstream vLLM v0.31.0**, plus a
quantized KV cache path that reaches a **271,925-token pool** at
`max_model_len 262144` - **6.18x** the fp16 KV capacity - and serves it at
**34-37 tok/s** single-stream with **~1,060 tok/s** prefill.

| headline | measured value |
|---|---|
| KV pool @ 262,144 context, util 0.90, 3-bit | **271,925 tokens** (6.18x fp16 KV) |
| Decode, 1 stream | **34.4-36.8 tok/s** (27.2 ms/step) |
| Prefill | **1,060 tok/s** @10.8k, 960 tok/s @48.6k |
| Quality, paired per-token NLL vs fp16 KV | 3-bit: +0.011 @8k, +0.0038 @32k; 4-bit: +0.001 @8k |
| Hardware | 2 x V100-SXM2-16GB, tensor-parallel 2, PCIe (no P2P, no NVLink) |

## What makes this different

- **Built on current upstream, not a fork.** Seven concern-split patches apply to
  the pinned vLLM **v0.31.0** tag, and the ported kernels register through the
  **stable ABI** (`_C_stable_libtorch`) - so they load into a modern vLLM
  instead of depending on a fork's central bindings. You keep upstream's tree,
  its API surface and its updates.
- **Quantized KV cache on hardware that has no FP8.** TurboQuant
  `4bit_nc` (effectively lossless) and `3bit_nc` (maximum capacity) run on Volta,
  where the caching trick everyone else relies on - fp8 KV - does not exist. The
  capacity ladder is measured, preset by preset: 44,092 fp16 -> 157,803 -> 220,880
  -> 237,240 -> **272,570 tokens**.
- **Long context on 32 GB of Volta.** 262,144 tokens is the model's native
  context, not extrapolation, and it fits with room for concurrent streams and
  CPU KV offload.
- **Quality is measured, not asserted.** Paired per-token NLL against an fp16-KV
  anchor on identical token ids, plus a decode step-cost model
  (`27.7 ms + 0.302 ms per 1k resident tokens`) you can size capacity from.
- **Every number is reproduced by a script that ships here.** The harnesses in
  `harnesses/` produced every figure in this file, and all throughput is counted
  by the **server** (`usage.*`) - never estimated from text length.
- **The limits are published too, with evidence.** `KNOWN-LIMITS.md` carries the
  measured defects (one KV preset is unusable on Volta and now fails loudly, not
  silently), what is measured *faster* but not measured *equal*, behaviour changes
  that affect non-Volta builds, and the regions that were not reviewed.
- **You can verify instead of trusting.** `verify_series.sh` re-applies the
  patches to upstream v0.31.0 and byte-compares every file; `sha256sum -c
  SHA256SUMS` checks what you received; `smoke_test.py` runs 11 checks against a
  live server, including a NaN-proof throughput reading and which decode path the
  engine actually chose. No binaries, no wheels, no model weights.

> **PREVIEW / experimental - not for production.** Read `KNOWN-LIMITS.md` before
> running anything, and do not pick a KV-cache dtype for a workload you care
> about without reading section 11 of it. The quality numbers in this file were
> measured on one configuration; the parts listed as unresolved in sections 9-12
> are unresolved - read those as claims that have not been verified, not as
> footnotes.

## Release identity

| | |
|---|---|
| project | **Long-vLLM** - the name of this port; it is what the repository, the
release archive and this document are called |
| release | **`v0.31.0-sm70-preview.1`** (preview / experimental) |
| date | 2026-11-05 |
| applies to | upstream vLLM **v0.31.0**, tag `db9527a46873454610df6dbedf79a36d6bf1a7f6` |
| contents | 7 patches (`patches/0000`-`0006`), 437 files, one status + provenance row per path in `MANIFEST.tsv` |
| ships | patches, reproduction harnesses and documents only - **no binaries, no wheels, no model weights**. You apply, build, and supply the model yourself |
| before you build | `verify_series.sh` reproduces the set byte-for-byte from the patches, against upstream v0.31.0 itself, and `sha256sum -c SHA256SUMS` confirms the files you received |
| one thing the checks cannot see | modes: `vllm/envs.py` is executable upstream and in this set, and a byte comparison would not notice if that were lost - after applying, `git ls-files -s vllm/envs.py` should print mode `100755` |
| after you build | `smoke_test.py` runs 11 checks against a live server |
| tested on | the configuration in the next table - and nowhere else |

## Validated configuration

| | |
|---|---|
| GPUs | 2 x V100-SXM2-16GB, tensor-parallel 2 |
| Interconnect | PCIe, **PHB, no P2P, no NVLink** (this matters - see limits) |
| Model | Qwen3.8-27B NVFP4 (64 layers: 48 gated-delta-net + 16 full attention, head_dim 256, native context 262,144) |
| Stack | Python 3.12, torch 2.10.0+cu128, CUDA 12.8, driver 580.x |
| Attention backend | `TURBOQUANT` (SM70 FlashAttention-V100 kernels) |

## What works

- **Quantization**: compressed-tensors NVFP4 (`W4A4Fp4`, `use_a16=True`),
  Marlin W4A16 / AWQ / GPTQ, TurboMind W4A16 dense, Marlin MoE.
- **Quantized KV cache**: `turboquant_3bit_nc` (smallest, ~12.4 KB/token),
  `turboquant_4bit_nc` (effectively lossless), `turboquant_k3v4_nc`.
- **Long context**: 262,144 tokens with 3-bit KV - this is the model's native
  limit, nothing is extrapolated.
- **CPU KV offload** (`--kv-offloading-backend native`), tensor parallel 2,
  chunked prefill, CUDA graph decode (`FULL_DECODE_ONLY`).
- **OpenAI API surface**: tool calls (`qwen3_xml`), reasoning (`qwen3`), and
  usage details (`--enable-prompt-tokens-details`).

## Measured results

All throughput figures are counted by the **server** (`usage.*`, `/metrics`),
never estimated from text length.

| measurement | value |
|---|---|
| Decode, 1 stream, short context | **36.8 tok/s** (27.2 ms/step) |
| Decode, 3 streams, ~1.1k resident | 29.4 tok/s/stream (74.6 tok/s aggregate) |
| Decode, 4 streams, 174k resident | 12.3 tok/s/stream (49.3 tok/s aggregate) |
| Prefill | **1,060 tok/s** @10.8k, **960 tok/s** @48.6k |
| KV pool @ `max_model_len 262144`, util 0.90, 3-bit | **271,925 tokens** |
| KV capacity vs fp16 | **6.18x** |

Decode step cost, fitted over pure-decode steps, is the single most useful
number for capacity planning:

    step_ms ~= 27.7 + 0.302 * (resident_tokens / 1000)

Because a step is shared by the whole batch, **adding concurrent streams does
not raise aggregate throughput** - it splits a ceiling set by the per-stream
context length. See [KNOWN-LIMITS.md](KNOWN-LIMITS.md#3-throughput-ceiling).

## Build

The build is in-source and needs the two vendored extensions to be importable.
Reproduce with `build_sm70.sh` (read it first - it is short and explicit):

```bash
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS=16
bash build_sm70.sh /path/to/vllm
```

Two requirements are enforced by the build, and one of them fails late:

- **`patchelf` must be on `PATH`.** `setup.py` strips the build host's RPATH from
  the bundled SM70 extensions and refuses to ship a wheel that leaks it
  (`remove_rpath`, raised after the CUDA compile has already finished). Upstream
  vLLM source builds do not need `patchelf`; this set does.
- **`TORCH_CUDA_ARCH_LIST` must be set.** SM70 targets are selected by it. With
  the variable unset the SM70 extensions are skipped **without any message** -
  the build succeeds and you get a vLLM with none of the kernels this set exists
  to provide. See [KNOWN-LIMITS.md](KNOWN-LIMITS.md#9-behaviour-changes-that-affect-users-who-are-not-on-volta).

Outline:

1. `cmake -S . -B .` in the vLLM tree. (Passing
   `-DVLLM_FLASH_ATTN_SM70=ON` is harmless but unnecessary: the build derives that
   variable from `TORCH_CUDA_ARCH_LIST`, so step 0 above is what actually selects
   SM70.)
2. `ninja _C_stable_libtorch.abi3.so` - the C++ extension.
3. Copy the built `_C_stable_libtorch.abi3.so` **into the `vllm/` package
   directory** (it is produced at the repo root, and that is where it is
   imported from).
4. Make the vendored extensions importable via `.pth` files in site-packages
   (`flash-attention-v100`, `flash_qla`) - see `build_sm70.sh`.

Step 3 is the one people trip over: the extension builds to the repo root but
is imported from the package directory.

**If a server is already running from this tree, do not overwrite that `.so` in
place.** `cp -f` onto a file a live process has mapped keeps the same inode, so
the running process page-faults into half-old, half-new code - which is a
segfault inside whichever kernel is executing, not a clean failure. Write to a
temporary name and `mv` it into place (atomic rename, new inode), then restart:
check with `grep -l _C_stable_libtorch /proc/*/maps` first.

## Serve

The environment variables below are part of the configuration that produced the
numbers in this file; the flags do not all default to these values.

```bash
export TORCH_CUDA_ARCH_LIST=7.0
# The Flash-V100 TurboQuant decode path is opt-in and needs a matching split
# count (see the note below). Without both lines, decode falls back to the
# slower Triton path.
export VLLM_SM70_TURBOQUANT_FLASH_V100_DECODE=1

vllm serve /path/to/model \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --kv-cache-dtype turboquant_3bit_nc \
  --mamba-ssm-cache-dtype float16 \
  --attention-backend TURBOQUANT \
  --max-model-len 262144 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.90 \
  --kv-offloading-size 64 \
  --kv-offloading-backend native \
  --enable-chunked-prefill \
  --enable-prompt-tokens-details \
  --attention-config '{"tq_max_kv_splits_for_cuda_graph":264}' \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}'
```

Notes on that command:

- `--kv-offloading-size 64` puts CPU KV offload in `/dev/shm`; do **not** set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` with it, and never combine
  it with `cpu_offload_gb` (see `KNOWN-LIMITS.md`).
- `--attention-config '{"tq_max_kv_splits_for_cuda_graph":264}'` is required by
  the Flash-V100 decode path: the kernel sizes its split tiles from the full
  capacity, so `N * 1024 >= max_seq_capacity`. `max_seq_capacity` rounds up to a
  whole KV block, so for 262,144 it is 264,192 and `N` must be at least 258.
  If the flag is left on without a sufficient `N`, the backend raises at startup
  and tells you the value it needs.
- Omit `VLLM_SM70_TURBOQUANT_FLASH_V100_DECODE` (or set it to 0) and the decode
  path falls back to Triton, which is correct but several times slower.

Environment knobs used by that configuration:

| variable | default | meaning |
|---|---|---|
| `VLLM_SM70_TURBOQUANT_FLASH_V100_DECODE` | `0` | Flash-V100 kernel reads the packed TQ cache directly instead of dequantizing it per step. Much faster; numerically unverified (see `KNOWN-LIMITS.md`). Needs the split count above. |
| `VLLM_SM70_GDN_{KKT,DELTA_H,CHUNK_O}_SCHEDULE` | `1` | pin the GDN Triton kernel schedules; without them the autotuner can OOM during warmup |
| `VLLM_SM70_GDN_FLASHQLA_PREFILL` | `1` | use the FlashQLA GDN prefill kernel |
| `VLLM_SM70_GEMMA_LONG_PREFILL_FUSED` | `1` | fused add + GemmaRMSNorm for long prefills |

Most booleans are parsed with `int()`, so set them to `0` or `1` — `=true`
raises. See `KNOWN-LIMITS.md` section 11 for the rest of the flag surface.

## Quality

Paired per-token NLL against an fp16-KV anchor on the same token ids (8k and
32k windows, teacher-forced):

| KV preset | NLL vs fp16 @8k | @32k | capacity |
|---|---|---|---|
| `turboquant_4bit_nc` | +0.000983 (z+0.6) | ~0 | 220,880 tok @ util 0.98 |
| `turboquant_k3v4_nc` | +0.000533 (z+0.3) | **-0.0036** (z-4.1) | 237,240 |
| `turboquant_3bit_nc` | +0.010740 (z+3.5) | +0.003839 (z+2.9) | **272,570** |
| `turboquant_k8v4` | **+0.4896 (z+29.5)** | **+0.708** | unusable, see limits |

The cost of 3-bit is almost entirely in the **values**; 3-bit keys are free
(Hadamard rotation spreads key error across head_dim=256 and the norm
correction restores ||k|| exactly).

Read the negative deltas as **no detectable loss, not as an improvement**.
`turboquant_k3v4_nc` at 32k measures *below* the fp16 anchor it replaces, and a
lossy cache cannot genuinely be better than the fp16 it is quantizing. The sign
is reproducible in this harness but its cause is not established, so it is
printed rather than rounded to zero: an unexplained sign is a fact about the
measurement, and the honest reading of any row here is "no loss this comparison
can resolve". `KNOWN-LIMITS.md` section 2 records what has and has not been
established about the decode path's numerics.

## Relationship to 1Cat-vLLM

This patch set ports SM70 kernel work from
[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) (Apache-2.0) onto current
upstream vLLM. Ported directories include `csrc/attention/sm70_79t`,
`csrc/attention/sm70_v37`, `csrc/attention/sm70_grouped_long`,
`csrc/sm70_turbomind`, the SM70 GDN schedule pins, and the SM70 long-prefill
fused add + RMSNorm kernel. Their copyright headers are retained where present.

**This is not a replacement for 1Cat-vLLM** and should not be read as one. If
you want their full stack, use their project. The delta here is: current
upstream base, stable-ABI op registration so ported kernels load in vLLM
v0.31.0, TurboQuant-on-Volta with measured quality numbers, and the harnesses
used to produce the numbers above.

See `NOTICE` for all attributions: 3 license families (Apache-2.0,
BSD-3-Clause, MIT) across 13 attribution entries, with the licence text for each
directory it names.

## Reproduction harnesses

Every number in this file was produced by a script in `harnesses/`, which ships
with this set precisely so the claims can be checked rather than believed. Read
`harnesses/README.md` for the claim-by-claim mapping; the short version:

| claim | harness |
|---|---|
| decode / prefill throughput, KV pool | `sm70_bench.py` |
| decode step-cost model (`27.7 + 0.302/1000` ms) | `sm70_decode_probe.py` |
| the quality table (paired per-token NLL) | `sm70_kv_nll.py`, `sm70_ppl.py` |
| capacity ladder | `sm70_kv_ladder.sh`, `sm70_kv_ladder_analyze.py` |
| op-level write/decode/prefill error | `sm70_kv_probe.py` |
| CPU KV offload: works, costs no GPU KV, lossless | `sm70_offload_check.sh`, `sm70_offload_quality.py` |
| interconnect evidence (PHB, no P2P) | `sm70_trace_summary.py` |
| why `turboquant_k8v4` fails on Volta | `sm70_fp8_roundtrip_probe.py` |
| the generated stable-ABI registry | `gen_sm70_stable_reg.py` |

They come in three kinds, and the distinction matters: **offline kernel probes**
(`sm70_kv_probe.py`, `sm70_decode_probe.py`, `sm70_tq_pad_probe.py`,
`sm70_fp8_roundtrip_probe.py`) need neither a server nor a model and are the
cheapest check that a build is sane; **in-process runs** (`sm70_bench.py`,
`sm70_kv_nll.py`, `sm70_ppl.py`, `sm70_flashqla_logprob.py`) construct `LLM(...)`
themselves and need a model directory; only the **CPU-KV-offload pair**
(`sm70_offload_check.sh`, `sm70_offload_quality.py`) talks to a running server,
via `VLLM_OQ_BASE`.

```bash
python3 harnesses/sm70_kv_probe.py                    # no server, no model
VLLM_NLL_MODEL=/path/to/model python3 harnesses/sm70_kv_nll.py
VLLM_OQ_BASE=http://127.0.0.1:30000 python3 harnesses/sm70_offload_quality.py record --out oq.json
```

