# Reproduction harnesses

Every measurement in `README.md` was produced by a script in this directory.
They are included because a preview whose numbers cannot be reproduced from the
artefact is not much use to a reader.

These are the scripts as they were run. Host-specific paths were replaced with
placeholders — `/path/to/model`, ``SM70_TREE``, `SM70_LAUNCH_A`/`SM70_LAUNCH_B` —
and every replacement is asserted at build time, so a pattern that stops
matching fails the build instead of shipping a local path. Nothing else was edited: a harness that
needed a real fix was not shipped until it was fixed and run.

## How they get their input

Three kinds, and it matters which one you are running:

- **offline kernel probes** — no server and no model. They import the ported
  kernels directly and compare them against a plain torch reference. These are
  the cheapest way to check that a build is sane.
- **in-process model runs** — they construct `LLM(...)` themselves, so they need
  a model directory (`VLLM_NLL_MODEL` / `VLLM_PPL_MODEL` / `VLLM_BM_MODEL`) and a
  corpus of token ids, and they occupy the GPUs while they run.
- **HTTP against a running server** — only the CPU-KV-offload pair does this,
  using `VLLM_OQ_BASE` (default `http://127.0.0.1:30000`). Point it at whichever
  server you have running.

There is no uniform endpoint variable: each harness takes its own input, listed
in the table below.

| claim it backs | harness | kind | interface |
|---|---|---|---|
| Decode and prefill throughput, KV pool size | `sm70_bench.py` | in-process | `VLLM_BM_MODEL`, `VLLM_BM_*` knobs (TP, kv dtype, maxlen, concurrency, cudagraph mode) |
| Decode step-cost model (`27.7 + 0.302/1000` ms) | `sm70_decode_probe.py` | offline | none — sets up a fused KV cache and times the decode op |
| Op-level error (write 0.0, decode 1.2e-4, prefill 6e-4) | `sm70_kv_probe.py` | offline | none — replicates the engine's KV path without the engine |
| Quality table (paired per-token NLL per KV preset) | `sm70_kv_nll.py` | in-process | `VLLM_NLL_MODEL`, `VLLM_NLL_CORPUS`, `VLLM_NLL_KV`, `VLLM_NLL_WINDOW`, `VLLM_NLL_MAXLEN` |
| Perplexity checks | `sm70_ppl.py` | in-process | `VLLM_PPL_MODEL`, `VLLM_PPL_TOKENS`, `VLLM_PPL_KV_DT`, `VLLM_PPL_MAXLEN` |
| KV capacity ladder (44,092 → 272,570 tokens) | `sm70_kv_ladder.sh` | driver + in-process | `argv: <tree> <outdir>`; runs one server per rung |
| …and its analysis | `sm70_kv_ladder_analyze.py` | file input | `argv: <nll-dir>` |
| CPU KV offload: works, costs no GPU KV, lossless | `sm70_offload_check.sh` | HTTP + launcher | `SM70_TREE`, `SM70_LAUNCH_A` (TurboQuant profile), `SM70_LAUNCH_B` (offload profile) |
| CPU KV offload quality under mixed load | `sm70_offload_quality.py` | HTTP | `VLLM_OQ_BASE`, `VLLM_OQ_MODEL`, `VLLM_OQ_TOKENIZER`, `VLLM_OQ_CORPUS`; modes `record` / `stress` / `compare` |
| Interconnect evidence (PHB, no P2P, NCCL share) | `sm70_trace_summary.py` | file input | `argv: <profile-trace-dir> [<top-n>]` |
| Short-continuation front padding | `sm70_tq_pad_probe.py` | offline | none — proves the FA2 `q_len >= 1024` requirement |
| Why `turboquant_k8v4` is unusable on Volta | `sm70_fp8_roundtrip_probe.py` | offline | none — the FP8 round-trip error on hardware without FP8 |
| FlashQLA GDN numeric check | `sm70_flashqla_logprob.py` | in-process | `VLLM_FQ_MODEL`, `argv: <corpus>` |
| The generated stable-ABI registry | `gen_sm70_stable_reg.py` | code generator | `--out`, `--ref`; `csrc/libtorch_stable/sm70_stable_registry.cpp` comes from this |

Runtime verification of a live server is at the top level: `smoke_test.py`
(11 checks, including that the log shows which decode path actually ran).
