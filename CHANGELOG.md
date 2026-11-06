# Changelog

## v0.31.0-sm70-preview.1 — 2026-11-05

First preview release. Applies to upstream vLLM **v0.31.0**
(`db9527a46873454610df6dbedf79a36d6bf1a7f6`). **Experimental, not for
production** — read `KNOWN-LIMITS.md` before using it for anything you care
about.

### What this is

- SM70 (Volta / V100) support ported onto current upstream vLLM, built on
  1Cat-vLLM's SM70 work. 7 patches, 437 files, `MANIFEST.tsv` giving each path a
  status and a provenance.
- Quantized KV cache on Volta (TurboQuant), with paired per-token NLL quality
  numbers and a measured capacity ladder, not just a "supported" claim.
- Stable-ABI op registration, so the ported kernels load in v0.31.0 instead of
  requiring 1Cat's non-stable central bindings.

### Verified in this release

- `_C_stable_libtorch` compiles, loads, and serves: 11/11 runtime checks in
  `smoke_test.py` on 2x V100-SXM2-16GB, no engine death, both workers on the
  Flash-V100 packed-cache decode path.
- Decode **34.4–36.8 tok/s** single stream and prefill ~1,060 tok/s, counted by
  the server (`usage.*`), never estimated from text length.
- KV pool **271,925 tokens** at `max_model_len 262144`, util 0.90, 3-bit KV.
- `turboquant_k8v4` now **fails loudly** on devices without FP8 (SM < 8.0) rather
  than silently decoding keys wrongly: measured +43% NLL at 8k, +57% at 32k.
- `verify_series.sh` reproduces every shipped file byte-identically from the
  patch series, and `sha256sum -c SHA256SUMS` verifies the archive you received.

### Fixed in this release

- `build_sm70.sh`: builds with
  `pip install -e . --no-build-isolation --no-deps`. The script used
  `setup.py develop`, which setuptools now redirects to an **isolated** PEP 517
  build; that would install `pyproject.toml`'s `torch == 2.13.0` instead of the
  venv's cu128 torch. `--no-build-isolation` alone was still not enough: plain
  `pip install -e .` also installs the project's runtime dependencies, so it
  replaced the venv's torch with PyPI's 2.13.0, which requires the CUDA 13
  runtime and on this driver cannot even be imported
  (`OSError: libcudart.so.13`).
- `build_sm70.sh`: no longer points `VLLM_FLASH_ATTN_SRC_DIR` at
  `flash-attention-v100`. That tree is the runtime package (it is exposed via a
  `.pth` file), not a drop-in for the CMake external project: aiming the variable
  at it removed both the FA3 sources and the CUTLASS example headers that
  `csrc/attention/sm70_v37/*.cu` include, so a clean build failed twice over -
  `ninja: error: unknown target '_vllm_fa3_C'`, then
  `gemm_with_epilogue_visitor.h: No such file or directory`.
- `setup.py`: the `_vllm_fa3_C` extension target is now requested only when the
  flash-attn tree the build will use actually has FA3 sources, so that mistake
  fails immediately and by name instead of after a long configure.
- `CMakeLists.txt`: the SM70 build now **generates** the marlin-MoE kernels it
  compiles. `csrc/moe/marlin_moe_wna16/ops.cu` includes `kernel_selector.h`, a
  generated and gitignored file, and upstream generates it only for its own
  `libtorch_stable` copy and only when the 7.5/8.0 arch intersection is
  non-empty. A clean Volta build therefore had no such header, while the
  `file(GLOB)` above it silently matched zero files. Returning the same archs
  reproduces the reference system's header byte for byte.
- **A boot-path guard could not boot.** It read `kv_cache_spec` from the
  attention *Impl*, which is never given one - every start in the documented
  configuration died with `AttributeError`. Now reads
  `cache_config.block_size`, and the split count it demands is checked against
  the real capacity.
- **`VLLM_SM70_TURBOQUANT_FLASH_V100_DECODE` defaults to `0`.** Left on without
  a matching `tq_max_kv_splits_for_cuda_graph`, the kernel raises on contexts
  above ~32k. The error now names the value it needs.
- **The `2000 tok/s` prefill headline was retracted.** It was a dashboard
  artifact plus a 4xV100 TP4 measurement, not a number from this configuration.
  `README.md` carries the corrected figures.

### Not shipped, and why

- Two `sm70_mtp_dynamic_vocab_*.pt` token-frequency assets: corpus-derived, their
  provenance cannot be documented, and nothing in the tree references them.
- `torch_patched_headers/`: a CMake build product (a copy of a torch header plus
  upstream's `cmake/patches/pytorch_stable_string.patch`), created only under
  torch 2.13.x. Our torch is 2.10, so it is not used here at all.

### Known limitations

Sections 1–12 of `KNOWN-LIMITS.md`. In short: the numbers come from one machine
and one model; `turboquant_k8v4` is unusable on Volta; the Flash-V100 decode path
is measured faster but not yet measured numerically equal to the Triton path; the
reported throughput ceiling is a property of GQA-redundant unpacking, not a knob.
