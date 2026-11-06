# Provenance

The Marlin W4A16/W8A16/W4A8 kernels in this directory are derived from
[IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) (Apache-2.0) and from
the Marlin kernels shipped by upstream vLLM (Apache-2.0). The `sm70_*` files are
additions made for this patch set: SM70 (Volta) code paths, tuned tile/mma
variants, and the NVFP4 and u8b128 specialisations. Everything here is
Apache-2.0; `LICENSE` is a copy of the licence text. Modifications for this port
are not individually annotated, so treat these files as a fork adaptively
maintained against upstream Marlin.
