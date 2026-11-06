# Known limits

Everything here was measured on the configuration in the README. Where a number
comes from a single machine, it says so. Read this before planning capacity.

## 1. `turboquant_k8v4` is unusable on SM70 - do not select it

`triton_turboquant_decode.py` decides whether to use an FP8 E4B15 cast with
`_use_fp8_e4b15() -> cap < (8, 9)`. That condition was written for
Ampere/Ada and **Volta (7.0) satisfies it**, so the kernel runs an FP8 cast on
hardware that has no FP8. Measured against an fp16-KV anchor, paired
per-token NLL:

| window | `turboquant_k8v4` |
|---|---|
| 8k | **+0.4896 NLL (z+29.5)**, ppl 3.15 -> 5.14 |
| 32k | **+0.708 NLL (z+71.8)** |

The degradation grows with context. Every other TurboQuant preset uses integer
codebooks and is unaffected.

**Use `turboquant_4bit_nc` (effectively lossless) or `turboquant_3bit_nc`
(~1% NLL, maximum capacity). Never `k8v4` on Volta.** This is a known defect in
this patch set, not fixed here: the safe minimum fix is to make
`_use_fp8_e4b15()` return 0 for `cap < (8, 0)` and to reject the preset in
`supports_kv_cache_dtype`.

## 2. The Flash-V100 decode path is measured faster, not measured equal

With `VLLM_SM70_TURBOQUANT_FLASH_V100_DECODE=1` the decode kernel reads the
packed TurboQuant cache in-kernel instead of dequantizing it into fp16 scratch
on every step (`turboquant_attn.py`: "skipping the O(seq_len) dequant that the
Triton kernel performs per step").

Measured effect, pure-decode steps only:

| load | Triton path | Flash-V100 path |
|---|---|---|
| 1 stream, short ctx | 128.2 ms/step | **27.2 ms/step** |
| 3 streams, ~14k each | - | 42.1 ms/step |
| 4 streams, 174k total | - | 81.1 ms/step |

What is **not** established: numeric equivalence with the reference path over
identical token ids. The API cannot teacher-force arbitrary tokens through the
decode kernel, so this needs an offline paired-NLL run with the flag on and off.
Until that is done, treat the flag as a performance option, not a default. Set
it to `0` to use the validated Triton path.

Related trap: the flag is silent when it falls back. Check the server log for
`TURBOQUANT decode using Flash-V100 packed-cache path` before believing any
speedup.

**An unresolved sign in the quality comparison.** `turboquant_k3v4_nc` measures
*below* the fp16-KV anchor at 32k (paired per-token NLL, -0.0036, z-4.1) and
`turboquant_4bit_nc` measures below it at 8k (-0.000983, z+0.6). A lossy cache
cannot genuinely beat the fp16 it replaces, so these deltas are not quality
gains - but the cause of the sign is not established either, and an earlier
attempt to repeat the comparison at a larger context produced an inversion of
the 3-bit/4-bit ordering rather than a resolution of it. Treat the absolute
magnitude as an upper bound on the loss, never the sign as a benefit.

## 3. Throughput ceiling

    step_ms ~= 27.7 + 0.302 * (resident_tokens / 1000)

The slope is per *resident token in the batch*, so aggregate decode throughput
asymptotes to roughly:

    aggregate_tok/s ~= 3311 / (tokens resident per stream, in thousands)

| conversation size | ceiling, regardless of concurrency |
|---|---|
| 8k | ~410 tok/s |
| 50k | ~66 tok/s |
| 128k | ~26 tok/s |
| 256k | **~13 tok/s** |

Adding streams does not raise this - it divides the same ceiling. If you need
interactive speed at 256k, this is the wrong hardware (see section 5). Where the
remaining cost goes: the decode kernel launches one block per *query* head
(`grid.y = num_attention_heads`) while `kv_head = head_idx / q_per_kv`, so for a
model with `q_per_kv = 6` the same packed KV rows are unpacked six times. That
is the largest known remaining inefficiency, and it is not fixed here.

## 4. Long contested prefills can still OOM the engine

`flash_attn_v100.py` has five `except torch.OutOfMemoryError` guards that
degrade instead of dying. `turboquant_attn.py` does **not** have them on all
allocation sites. A long continuation prefill over a large cached prefix, run
concurrently with several other requests, can still raise CUDA OOM and take the
engine down.

Mitigations used here: keep `--max-num-batched-tokens` modest (2048), do not
launch many simultaneous long prefills, and keep `--gpu-memory-utilization`
below 0.95. The pad-to-1024 fix (`_FA2_MIN_QLEN`) removed the worst transient -
the SM70 FA2 kernel hard-requires `q_len >= 1024`, and front-padding to it is
numerically exact - but the remaining allocation sites are unguarded.

## 5. Hardware facts that no patch can change

- **No FP8 on Volta.** fp8 KV cache is not available; capacity comes from
  TurboQuant integer presets instead.
- **No P2P and no NVLink** on the reference system (PCIe, PHB). Tensor-parallel
  all-reduce is host-staged through NCCL at ~3.6 GB/s, which makes prefill
  interconnect-bound: measured ceiling ~1,470 tok/s at TP2. Machines with
  NVLink will do better; this number is not a property of the patch set.
- Volta has no `cp.async`, no TMA, no FP8, no `bf16` tensor-core path of the
  Ampere kind. Kernels written for those must be adapted, not just recompiled.

## 6. CPU KV offload: what it does and does not cost

- It costs **zero** GPU KV memory. A matched control (offload on vs off at the
  same `--gpu-memory-utilization`) is byte-identical in every memory line.
- The entire pool difference between a 0.98 and a 0.90 profile is the
  **utilization budget**, not the connector - it is headroom for continuation
  prefills.
- The offload region is an mmap file in `/dev/shm`, i.e. host RAM, not disk and
  not pinned.
- Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: the offloading
  connector rejects it unless the cumem allocator is enabled. It is a hard
  startup error, not a warning.

## 7. Cache-read reporting is block-granular

With hybrid KV groups the block size is the LCM of the group sizes (2064 tokens
in the reference config). `usage.prompt_tokens_details.cached_tokens` is
therefore always a multiple of the block size: a 1,837-token prompt reports 0,
a 6,388-token prompt reports 6,192. That is geometry, not a bug - but it will
make small-prompt cache hit rates look broken.

`--enable-prompt-tokens-details` is required for that field to be populated at
all; without it the field is `null` and every client reads 0%.

## 8. Not implemented

- **Speculative decoding / MTP.** The checkpoint used here ships no draft head.
  (The `vllm/assets/*.pt` files in this tree are token-id ranking metadata, not
  draft weights - see `PROVENANCE.md`.)
- **fp8 KV cache** on SM70 (see section 5).
- The published numbers were not reproduced on SM75/SM80/SM86, and SM70-specific
  code paths are gated so other architectures keep upstream behavior - but that
  gating is the highest-value thing for a reviewer to check in the diff.

## 9. Behaviour changes that affect users who are not on Volta

These were found by a review of the diff against upstream v0.31.0 and are **not
fixed** in this revision. Read them before deploying on non-SM70 hardware.

- **TP4/TP8 hierarchical all-reduce kernels spin without a timeout.** The push
  protocol polls a peer signal with no deadline, no watchdog and no
  capture-status guard in the up-mix path. A peer process that dies mid-step
  hangs the decode loop rather than failing. Only these configurations use it;
  the validated TP2 configuration does not.
- **`qwen38_hc_*` kernels bake in one model's shape.** They assert exact tensor
  sizes (`336`, `10240*320`, `10240`, `2560`) and map the rank through a
  four-way ternary with no bounds check. Another checkpoint is rejected by the
  size assertion or indexes out of range.
- **The push protocol reserves a 16-bit sentinel and silently rewrites any
  payload value matching it.** A user whose activations hit that pattern gets
  altered values, not an error.
- **A flash-attn external project is skipped silently if your arch set
  intersects neither `8.0+PTX` nor `7.0`** (for example a Turing 7.5-only
  build): a configure-time `STATUS` line is the only notice, and FlashAttention
  is missing at runtime.
- **A mixed `7.0`+`8.0` build drops all SM70 Marlin kernels** with a `STATUS`
  message.
- **`VLLM_SM70_PROFILE_TRACE` debug scaffolding is compiled in** (off by
  default) and writes to `std::cerr` from the all-reduce path.
- **This patch set targets `TORCH_TARGET_VERSION` 2.10**, lowered from upstream's
  2.11 because the port is validated on torch 2.10.0+cu128. If you build on
  torch >= 2.11 you may raise it back as noted in `CMakeLists.txt`.

- **The SM70 build is opt-in through an environment variable, and the opt-out is
  silent.** `_cuda_arches_from_env` (`setup.py:1391`) returns `None` when
  `TORCH_CUDA_ARCH_LIST` is unset, and `_cuda_arch_contains(7, 0)`
  (`setup.py:1426`) is then `False`, so no `vllm._sm70_*` target is registered and
  the FlashQLA extension is never built. Note the deliberate-looking asymmetry:
  `_cuda_arch_at_least(8, 0)` (`setup.py:1419`) returns **True** when unset, i.e.
  upstream behaviour degrades to "assume new", while the ported features degrade
  to "excluded". The parse is also all-or-nothing: a single token that does not
  match `(\d+)(?:\.(\d+))?`, e.g. `9.0a` in `7.0;9.0a`, nulls the whole list and
  silently drops SM70 even though 7.0 was requested. `build_sm70.sh` and
  `README.md` both set the variable, so the validated path is unaffected; a
  reader who builds the way upstream is usually built (`pip install -e .`) is
  not told anything is missing.
- **`bundle_flash_attn_v100` has no architecture gate** (`setup.py:215`, called at
  `setup.py:533` from a bare `if _is_cuda():`). Every CUDA build - including one
  on sm_80/sm_90 - recursively builds the vendored V100 tree with
  `TORCH_CUDA_ARCH_LIST=7.0` forced, and any failure is fatal
  (`raise RuntimeError(f"Failed to build Flash-V100 extension: ...")`). Its
  sibling `bundle_flash_qla_sm70` (`setup.py:272`) does carry the
  `_cuda_arch_contains(7, 0)` guard, so the asymmetry looks unintended.
- **The FlashQLA GDN extension is compiled for `sm_70` only, with no PTX**
  (`-gencode=arch=compute_70,code=sm_70`, `setup.py:294`) and is then `load()`ed
  into the build process with `with_cuda=True`. On a build host of sm_80 or
  newer with a mixed architecture list, that load fails and aborts the build
  rather than deferring the error to runtime. This is the mixed-architecture
  class already noted below; here it is in the build recipe.
- **`VLLM_USE_PRECOMPILED=1` produces a wheel with no SM70 extensions and no
  error.** The precompiled path (`precompiled_build_ext.run`, `setup.py:613`)
  calls `bundle_precompiled_flash_attn_v100`, which `return`s silently when the
  extracted tree lacks the extensions (`setup.py:249`), and it never calls
  `bundle_flash_qla_sm70` at all. The source path raises where this path is
  silent. The failure surfaces later as an import error at model load.
- **Wheel bundling of the two standalone extensions is inert.** `pyproject.toml`
  declares `[tool.setuptools.packages.find] include = ["vllm*"]`, so the
  `package_data` keys `flash_qla` and `flash_attn_v100` (`setup.py:1638`) refer to
  packages that are never declared of the distribution. This is consistent with
  how this tree exposes the vendored extensions - through a `.pth` pointing at
  the source tree - and it means a *wheel* built from this set should not be
  expected to contain them.

None of the four behavioural items above is fixed here: each would change what
someone else's build compiles, and there is no sm_80/sm_90 device on hand to test
the change on. They are documented so the choice is explicit rather than
accidental.

- **The ported long-prefill RMSNorm kernel does not compile for ROCm.**
  `layernorm_kernels.cu` wraps the kernel and its `kSm70GemmaHiddenSize` constant
  in `#ifndef USE_ROCM` (guard at `:219`, closed immediately before
  `}  // namespace vllm` at `:321`), but the host wrapper
  `sm70_gemma_long_prefill_fused_add_rms_norm` is *outside* the guard (`:480`)
  and references the guarded constant at `:493` and `:505`. A ROCm build
  therefore fails to compile rather than degrading. This set targets CUDA/SM70
  only, and no ROCm device exists here to test a fix on.

- **The SM70 collective wrappers resolve ops through two different namespaces.**
  `vllm/_custom_ops.py` has both `_custom_ar_op(name)`, which prefers an opt-in
  sidecar library (`torch.ops._C_custom_ar_flashnext`) and falls back to
  `_C_custom_ar`, and `_custom_ar_owner_namespace()`, whose comment says the
  opaque communicator belongs to the DSO that initialised it and that an
  optional op "must not fall through to another DSO with a different ABI". The
  handle-taking wrappers are split between the two: `sm70_tp2/tp4_*`,
  `all_reduce_sum2`, `top1_argmax` and `sm70_tp4_push_allreduce_buffer_size` use
  the sidecar-preferring lookup, while the `sm70_qwen38_hc_*` and
  `sm70_tp8_hierarchical_push_allreduce_buffer_size` wrappers use the owner
  namespace. Since `init_custom_ar` always creates the handle in `_C_custom_ar`,
  a sidecar that also defines those ops would be handed a communicator from a
  different DSO. This is latent in the validated configuration, which sets no
  `VLLM_SM70_CUSTOM_AR_LIBRARY`, and then both lookups resolve to `_C_custom_ar`.
- **The sidecar library is loaded at import time with no error handling.**
  `_custom_ops.py` calls `torch.ops.load_library(os.getenv(...))` at module
  scope, so a stale path turns every vLLM start into an opaque loader error that
  never names `VLLM_SM70_CUSTOM_AR_LIBRARY`. Unreachable unless the variable is set.
- **The SM70 collective wrappers have no availability probe.** Unlike
  `sm70_marlin_available`, which tests with `hasattr`, these raise a bare
  `AttributeError` at first use when the sidecar is absent, rather than naming
  what is missing.
- **`validate_cu_seqlens=False` with a caller-supplied bound.**
  `qwen_gdn_linear_attn.py` passes `cu_seqlens=cu_arg` (from
  `attn_metadata.prefill_query_start_loc`) with the kernel's own validation
  disabled. Dropping the per-call host validation is a deliberate cost
  trade-off, but it also removes the only guard on that input: metadata that is
  non-monotonic or oversized becomes an out-of-bounds GPU access instead of an
  exception.
- **The vendored `flash_qla` import path is added to `sys.path` process-wide.**
  The layer appends the repository root to `sys.path` when the vendored tree is
  not importable, which also makes `tests/`, `benchmarks/`, `scripts/` and
  `csrc/` importable names for the rest of the process. The comment explains the
  constraint (the vendored tree sits beside `vllm/`); the entry is never removed.

**A latent ceiling on the GDN fast path.** `_sm70_flashqla_prefill_available`
gates on `head_k_dim == 128` and `float16`, so a model with a different head
dimension or in bf16 silently falls back to Triton. The fallback is a
performance loss, never a correctness one, because Triton is the default path.


- **An inert legacy registration fragment, inherited from the port.** The patch
  set adds 95 lines to `csrc/torch_bindings.cpp` registering the SM70
  custom-all-reduce ops in a legacy `TORCH_LIBRARY_FRAGMENT`. That file is
  compiled **only for ROCm** - its own comment says "Legacy _C extension (ROCm
  only - CUDA ops migrated to _C_stable_libtorch)" and CMakeLists.txt gates it
  with `if(VLLM_GPU_LANG STREQUAL "HIP")` - and the added block is itself wrapped
  in `#ifndef USE_ROCM`. So the block is excluded in the one configuration that
  would compile it: it cannot run in any build, and the "registered twice in one
  namespace" hazard it looks like cannot arise. Recorded here because dead
  registrations mislead a reader into thinking that file still binds CUDA ops.
- **One MoE op is registered twice inside a single binary, and torch silently
  keeps the last one - a duplicate introduced by this patch set.** Loading
  `_C_stable_libtorch.abi3.so` prints torch's "Overriding a previously registered
  kernel for the same operator and the same dispatch key" for
  `_C_stable_libtorch::moe_wna16_marlin_gemm`. Upstream v0.31.0 keeps that code
  under `csrc/libtorch_stable/moe/marlin_moe_wna16/`; this set *adds* a parallel
  copy under `csrc/moe/marlin_moe_wna16/` (`ops.cu`, `kernel.h`,
  `marlin_template.h`, `generate_kernels.py`, and the SM70 files), and **both**
  registration sites are additions of this set - `sm70_marlin_moe_dispatch.cu:659`
  and `ops.cu:878` - compiled into the same DSO. Torch keeps whichever registered
  last (it names `ops.cu:877` in this build), so which implementation serves the
  op depends on static-initialisation order across translation units, which the
  language does not specify. The validated configuration cannot exercise it: the
  model is dense (48 GDN + 16 attention layers, no MoE) and there is no MoE model
  on Volta here to test the ordering with. Resolving it means making the two
  registrations mutually exclusive and re-testing an MoE model on SM70 - work for
  a later revision, recorded rather than guessed at.

- **The fused-norm GDN route is not on the validated path, so its aliasing is
  unverified.** `forward_cuda` chooses between two custom ops with
  `enable_fused_gdn_decode`, which is `gdn_decode_kernel == "cuda"`; the running
  service reports `GDN decode kernel: triton`, so the route through
  `qwen_gdn_attention_core_fused_norm_packed` -> `_forward_core_fused_norm` is not
  exercised by the headline validation in this document. In that route
  `_rms_norm_gated_cuda` is called with `core_attn_out[:num_actual_tokens]` as both
  `x` and `out`, i.e. it relies on the vendored `layer_norm_fwd` being
  in-place-safe. Upstream FLA does use it in place and the three shape asserts
  above the call do not check for aliasing, so this is probably fine - but
  "probably fine" is exactly what this section is for: a reader who enables the
  fused CUDA GDN path on SM70 would be the first to exercise it, and it has not
  been checked against a paired-NLL baseline here. `_forward_core_fused_norm` also
  calls `b.contiguous()`/`a.contiguous()` on every decode step in that route
  (performance, not correctness).
- **One argument is declared mutated although the standard path only reads it.**
  `qwen_gdn_attention_core` is registered with
  `mutates_args=["a_or_z_out", "core_attn_out"]`, while its own docstring says
  `a_or_z_out` is `a` and read-only unless `use_aiter=True` (ROCm), where it is
  the `z` output. Declaring a read-only argument as mutated is the conservative
  direction - the compiler must materialise a copy and cannot reuse the buffer
  across graph boundaries - so this is a small cost, not a wrong result, and it is
  correct for the ROCm path. The sibling op
  `qwen_gdn_attention_core_fused_norm_packed` declares only `core_attn_out`, which
  matches what it does.

- **Fixed during publication review, recorded because it was shipped broken for a
  while: the FA3 extension target on a Volta build.** `setup.py` appended the
  `_vllm_fa3_C` extension whenever CUDA was 12.3 or newer, but the CMake external
  project creates that target only when the flash-attn tree has FA3 sources.
  `build_sm70.sh` used to set
  `VLLM_FLASH_ATTN_SRC_DIR=$TREE/flash-attention-v100`, which has no `hopper/`
  directory, so the build died in `cmake --build` with `ninja: error: unknown
  target '_vllm_fa3_C'`, *after* a successful configure, which makes it look like
  a toolchain problem. The same override also removed the CUTLASS example headers
  that `csrc/attention/sm70_v37/*.cu` include, so `_vllm_fa2_C` then failed with
  `gemm_with_epilogue_visitor.h: No such file or directory`. Both are fixed at
  the root: the script no longer points that variable at the Volta tree (which is
  the runtime package, exposed via `.pth`), and `setup.py` additionally guards the
  FA3 request on the tree actually having FA3 sources. This was found by building
  the published set from a clean checkout rather than from the tree that happened
  to be on the machine - the machine's configure had used upstream's flash-attn
  tree, and it never built `_vllm_fa2_C` at all, so neither failure could appear
  there.

- **The documented build must pass `--no-build-isolation`, and `build_sm70.sh`
  now does.** `pyproject.toml`'s `[build-system]` pins `torch == 2.13.0`, while
  this port runs the venv's own cu128 torch (2.10.0 on the reference system).
  setuptools' `develop` command is deprecated and delegates to
  `pip install -e . --use-pep517`, and an isolated PEP 517 build installs those
  build requirements into a scratch environment - so it tries to fetch a torch
  build this driver cannot use, instead of the one sitting in the venv. Dropping
  isolation is not enough on its own: plain `pip install -e .` also installs the
  project's **runtime** dependencies, so it replaces the venv's torch with
  PyPI's `torch 2.13.0`, which wants the CUDA 13 runtime and on a driver-580 /
  CUDA-12.8 host cannot even be imported (`OSError: libcudart.so.13`). The script
  therefore builds with `pip install -e . --no-build-isolation --no-deps`. Anyone
  porting this to a similar host should keep both flags and make sure the venv
  holds the matching torch. The extension itself is stable-ABI, so a binary built
  with the wrong torch headers still loaded and registered its ops against the
  venv's torch - but that is luck to rely on, not a design.

- **Fixed during publication review: generated marlin-MoE sources were never
  generated, so a clean checkout could not build.** `csrc/moe/marlin_moe_wna16/`
  is a copy of upstream's tree, and like upstream's it gitignores the generated
  `kernel_selector.h` and `sm*_kernel_*.cu`. Upstream's CMake generates them for
  its `libtorch_stable` copy and only when the 7.5/8.0 arch intersection is
  non-empty; the port compiles the `csrc/moe` copy, for which nothing generated
  anything. On the reference system the files happened to be present, so the
  build worked and the published set did not contain them. A clean checkout failed
  in `ops.cu` with `fatal error: kernel_selector.h: No such file or directory`
  - after the `file(GLOB)` above it had silently matched zero files. The CMake now
  runs that directory's `generate_kernels.py` before the globs, with the same
  hash-and-arch cache guard upstream uses. Returning the same archs (7.5, 8.0)
  reproduces the reference system's `kernel_selector.h` byte for byte
  (sha256 `0d9127269992f44a...`), which is how the fix was checked.

## 10. Regions not verified here

This patch set was not reviewed line by line everywhere. The region below was
**not** read in full and should be reviewed before you rely on it:

- `vllm/_custom_ops.py` lines **1-4711**. The part this set actually touches
  (4714-4951) was read; the rest of that file is untouched upstream.

Everything else this set modifies in Python was read in full: `vllm/envs.py`,
both protocol serializers, `qwen_gdn_linear_attn.py`, `setup.py`, the two
`torch_bindings.cpp` files, `layernorm_kernels.cu` and `ops.h`.

The numbers in `README.md` come from the validated TP2 configuration and were
not re-measured after the last fixes to the build script.

## 11. Environment variables: read this before setting any

- **Use `0` and `1`, not `true`/`false`.** Most `VLLM_SM70_*` / `VLLM_FLASH_V100_*`
  flags are parsed as `bool(int(os.getenv(...)))`, so `VLLM_SM70_AWQ_TURBOMIND=true`
  raises `ValueError: invalid literal for int() with base 10: 'true'`. A minority
  accept `true/yes/on`; the split is not documented per flag.
- **A malformed value fails the whole service at startup, not just the flag.**
  `enable_envs_cache()` evaluates every registered key once at service init, so a
  bad value in a flag you never intended to use still aborts startup.
- **The flag surface is large and partly model-specific.** 239 `VLLM_SM70_*` /
  `VLLM_FLASH_V100_*` keys, of which 79 default to on. Many are named for the
  specific checkpoints they were tuned on (`QWEN38`, `QWEN3NEXT`, `GLM53`,
  `DSV4`, `DFLASH2`). Only one configuration is validated (see `README.md`).
  A flag name is not a promise that its behaviour transfers.
- **The Flash-V100 TurboQuant decode path requires a matching split count.**
  `VLLM_SM70_TURBOQUANT_FLASH_V100_DECODE=1` needs
  `--attention-config '{"tq_max_kv_splits_for_cuda_graph": N}'` with
  `N >= ceil(max_seq_capacity / 1024)` — 258 at a 256k context, against a default
  of 32. The backend now raises a named error at init with the exact value
  required. It is off by default (see `README.md` for the validated pair).
- Some flags are raw `os.getenv` reads that were not registered, so they are
  absent from generated documentation. Two of those are now registered; others
  may remain, and `envs.<NAME>` on an unregistered name raises `AttributeError`.


**Two ways to read a flag, one of which is invisible to the documentation.**
138 names are read as `envs.NAME`, and all 138 are registered, so they appear in
the generated documentation and raise `AttributeError` if misspelled. A further
478 names are read through raw `os.getenv("VLLM_...")` in the ported code; those
work when set but are absent from the generated flag list, so they are not
discoverable from the documentation.

## 12. Hard limits inherited from the torch 2.10 stable ABI and the SM70 kernels

- **At most 8 all-reduce ranks.** Upstream vLLM allows 16
  (`vllm::kMaxCustomCollectiveRanks`), but that constant's header is not
  reachable from this tree's stable-ABI translation unit and the SM70
  push/hierarchical kernels are sized for 4 and 8 ranks
  (`kSm70Tp4PushAllreduceWorldSize`, `kSm70Tp8HierarchicalPushWorldSize`). TP > 8
  throws at startup. Restoring 16 requires resizing the signal arrays in
  `csrc/libtorch_stable/custom_all_reduce.cuh` too.
- **`get_cuda_view_from_cpu_tensor` cannot release memory before process exit.**
  The torch 2.10 stable ABI's `from_blob` takes no deleter, so the pinned CPU
  tensors and the `cudaHostAlloc` staging buffers are held in a mutex-guarded
  registry until exit (only the raw host buffers are freed, from an atexit
  handler that makes no torch calls). If you offload with a large number of
  distinct unpinned CPU tensors, expect pinned-host growth for the process
  lifetime. This is the path the CPU KV offload uses; it is not bit-lossless.

**Kernel-level limits and assumptions in the ported long-prefill RMSNorm**
(`csrc/libtorch_stable/layernorm_kernels.cu`), found in the fourth review round
and left as found:

- **Row indexing overflows `int` at 1,677,722 rows.** The kernel computes
  `const int vector_row_offset = blockIdx.x * kVectorsPerRow;` (`:266`) with
  `kVectorsPerRow = 5120 / 4 = 1280`, and the wrapper's only bound on the batch
  dimension is `input.size(0) >= 256` (`:494`). Past that many rows the product
  wraps and the kernel reads and writes out of range. Unreachable in the
  validated configuration - rows are sequence positions, and the limit is
  262,144 - but it is a hard limit, not a checked one.
- **Pointer alignment is assumed, not checked.** The kernel reinterprets the
  half input as an 8-byte `Sm70Half4` and the fp32 residual as a 16-byte
  `float4`, while the wrapper checks dtype, shape, device and `is_contiguous()`
  for all five tensors but never alignment - unlike upstream's
  `fused_add_rms_norm`, which computes `ptrs_are_aligned` against
  `req_alignment_bytes`. This is safe here only because the contiguity checks
  plus `hidden_size = 5120` make every row stride a multiple of 16 bytes and
  torch's allocations are 256-byte aligned. A future hidden size that is not a
  multiple of 4 halves, or a caller passing an unaligned view, would not be
  caught.
- **`ops.h` includes less than it uses.** It uses `std::tuple` and `int64_t`
  while including only `<optional>`, `<string>` and `<vector>` (`:7`-`:9`); it
  compiles through torch's headers including those transitively. A leaner or
  reordered torch header set would break the build.

