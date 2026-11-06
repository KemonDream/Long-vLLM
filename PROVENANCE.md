# Provenance

Where the non-upstream parts of this patch set come from, and what they are.
`NOTICE` carries the license obligations; this file says what is in the tree.

## Vendored / ported trees

| path | origin | license | notes |
|---|---|---|---|
| `csrc/attention/sm70_79t/` | 1Cat-vLLM | Apache-2.0, plus CUTLASS files | CUTLASS headers keep their per-file BSD-3 notices |
| `csrc/attention/sm70_v37/` | 1Cat-vLLM | Apache-2.0 | SPDX headers present |
| `csrc/attention/sm70_grouped_long/` | D.Skryabin, via 1Cat-vLLM | **BSD-3-Clause** | `LICENSE` in the directory; SPDX headers added to each source file by this patch set |
| `csrc/sm70_turbomind/` | 1Cat-vLLM + LMDeploy + v100-skinny | Apache-2.0 / MIT | `ops/LICENSE.v100-skinny` retained; `lmdeploy/LICENSE` added by this patch set |
| `csrc/sm70_turbomind/lmdeploy/` | LMDeploy (TurboMind) | Apache-2.0 | subset; see `PROVENANCE.md` in that directory |
| `flash-attention-v100/` | D.Skryabin | **BSD-3-Clause** | vendored tree; `LICENSE` and `AUTHORS` retained |
| `flash_qla/` | Qwen / Alibaba Group, via 1Cat-vLLM | **MIT** | `LICENSE` retained, per-file headers retained |
| `flashinfer-sm70/` | written for this patch set | Apache-2.0 | **not** vendored FlashInfer: no FlashInfer source is copied. It provides a small Volta WMMA fragment layer under a FlashInfer-shaped include path (`include/flashinfer/attention/sm70/`) and is compiled directly by `CMakeLists.txt`. |
| `csrc/quantization/marlin/` | upstream vLLM, adapted from IST-DASLab/marlin | Apache-2.0 | `sm70_*` files are new for this patch set |
| `vllm/third_party/flash_linear_attention/` | FLA (Songlin Yang, Yu Zhang) | MIT | schedule pins added for SM70 |
| ~~`torch_patched_headers/`~~ | - | - | **excluded:** a CMake build product (copy of a torch header plus `cmake/patches/pytorch_stable_string.patch`), generated only under torch 2.13.x |

## Binary assets

None. Two files were excluded from this published set:

`vllm/assets/sm70_mtp_dynamic_vocab_qwen36_27b_{tp2,tp4}.pt` (about 1.9 MB each)
are integer token-id rankings plus SHA-256 fingerprints of token-frequency
statistics. They contain no weights and no text, but they are derived from a text
corpus whose provenance this patch set cannot document, and nothing in the tree
references them (the dynamic-vocabulary MTP path they belong to is not enabled in
the validated configuration). Rather than publish corpus-derived artefacts that
cannot be accounted for, they are omitted. Everything in this set is source code
and licence text.

## Relationship to 1Cat-vLLM

The SM70 kernel work originates in
[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) (Apache-2.0), pinned at commit
`e501a0f` while porting. This patch set's contribution is the port onto upstream
vLLM v0.31.0, the stable-ABI op registration that lets those kernels load there,
TurboQuant KV support on Volta with measured quality numbers, and the harnesses
that produced the numbers in `README.md`.

## Files removed by this patch set

`csrc/custom_all_reduce.cuh` (upstream, legacy host-side all-reduce header) is
deleted. The SM70 path replaces it with `csrc/libtorch_stable/custom_all_reduce.cu`
plus `csrc/libtorch_stable/custom_all_reduce.cuh`, which are built into the
stable-ABI extension. The only in-tree consumer of the old header was
`csrc/custom_all_reduce_test.cu`, a host-side test that no build target compiles;
it references the removed header and is therefore **not** part of this published
set.

## The snapshot the series was generated against, and why the reader's tree is safe

The series is generated as a diff between this repository's snapshot commit
`3a7a0bb` and the ported tree. That snapshot is *nearly*, but not exactly, the
upstream v0.31.0 tree. Measured tree-to-tree against the real tag
(`db9527a46873454610df6dbedf79a36d6bf1a7f6`):

- **7422 paths are common, and 0 of them differ in content.** Because git blob
  ids are content-addressed, the `index <old>..<new>` line in every patch for a
  modified file names the *same* preimage blob upstream's tree contains. That is
  why the series applies to a real v0.31.0 checkout rather than merely to this
  repository: 0 of the 401 added paths exist upstream, all 36 modified and the
  1 deleted path are present there.
- **The snapshot omits 9 upstream files**, which the series therefore does not
  mention and does not delete: `.claude/skills/{ci-fails-buildkite,debug-ima,
  kernel-microbenchmark,triton-kernel-writing}`, two benchmark JSONs
  (`benchmarks/multi_turn/generate_multi_turn.json`,
  `benchmarks/structured_schemas/structured_schema_1.json`), two tests
  (`tests/kernels/quantization/test_aiter_hipb_mm_linear_kernel.py`,
  `tests/v1/simple_kv_offload/test_hip_mem_ops.py`) and
  `vllm/vllm_flash_attn/.gitkeep`. A reader applying the series to v0.31.0 keeps
  those files and gets every file this set ships.
- **The snapshot adds 2 build artifacts** (`PKG-INFO`, `setup.cfg`), which are
  excluded from the set as artifacts.

Every status in `MANIFEST.tsv` was validated against the file list of the **real**
v0.31.0 tag tree (7431 paths), not against the snapshot. That distinction
matters: validating against the snapshot is self-consistent, so it cannot see
the difference.
