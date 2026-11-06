# FlashInfer SM70 - no patch series is shipped

This directory is intentionally empty of patches. No preparation script for a
vendored FlashInfer checkout exists anywhere in this tree. An earlier draft of this tree described an intended patch series
against a pinned FlashInfer v0.6.13 source; that work was **not** carried out and
is not part of this patch set. Nothing here should be read as a claim that a
FlashInfer attention backend runs on Volta.

What actually exists is smaller and self-contained:

* `../include/flashinfer/attention/sm70/volta_mma.cuh` - Volta WMMA fragment
  primitives, compiled for `sm_70`.
* `../csrc/h3_noncausal_sm70.cu` - one kernel entry point.
* `CMakeLists.txt` compiles the second file directly into the
  `_h3_flashinfer_C` extension, with `flashinfer-sm70/include` on the include
  path. No external FlashInfer checkout is required or used.

See `../README.md` for what the primitives do and what was measured, and
`../../KNOWN-LIMITS.md` for what does not work on Volta.
