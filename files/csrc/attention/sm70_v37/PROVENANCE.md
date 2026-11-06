# Provenance

The prefill attention kernels in this directory are ported from
[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) (Apache-2.0), pinned at commit
`e501a0f` while porting, into the upstream vLLM v0.31.0 tree.

- Licence: Apache-2.0. `LICENSE` in this directory is a copy of the licence
  text; every source file also carries its original copyright/SPDX notice.
- `include/cutlass/` headers come from NVIDIA CUTLASS (BSD-3-Clause) and keep
  their per-file notices.
- This patch set adapts the surrounding build integration and the op
  registration; the kernel sources themselves are 1Cat-vLLM's work. See the
  repository `NOTICE` for full attribution.
