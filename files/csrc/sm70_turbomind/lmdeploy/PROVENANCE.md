# Provenance

This directory contains a subset of LMDeploy's TurboMind sources, used to build
the SM70 (Volta) quantized GEMM path.

  Upstream : https://github.com/InternLM/lmdeploy  (TurboMind)
  License  : Apache License 2.0 - see LICENSE in this directory
  Note     : only the sources required by the SM70 GEMM kernels are vendored;
             unmodified where possible, and adapted where Volta lacks a feature
             (for example the SM90 TMA/GMMA headers are present but unused).

Adaptations made for this patch set are recorded in the commit history.
