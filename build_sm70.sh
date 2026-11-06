#!/usr/bin/env bash
# Build the SM70/V100 port of vLLM.
#
# Usage:  bash build_sm70.sh /path/to/vllm-checkout [/path/to/venv]
#
# This is a transcription of the build that produced the numbers in README.md.
# It is deliberately explicit; read it before running it. It does download:
# Python dependencies into the venv (numpy, pyyaml, ninja, setuptools-rust, ...,
# and torch if TORCH_WHL is set), plus the CMake dependencies that upstream vLLM
# fetches anyway - CUTLASS and triton-kernels, two large git clones, unless you
# seed them with VLLM_CUTLASS_SRC_DIR / TRITON_KERNELS_SRC_DIR below.
set -euo pipefail

TREE="${1:?usage: build_sm70.sh /path/to/vllm-checkout [venv-dir]}"
VENV="${2:-$TREE/../venv}"

export TORCH_CUDA_ARCH_LIST="7.0"        # Volta. Required; without it you get no SM70 cubins.
export VLLM_TARGET_DEVICE=cuda
export MAX_JOBS="${MAX_JOBS:-16}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# Do NOT point VLLM_FLASH_ATTN_SRC_DIR at $TREE/flash-attention-v100. That tree is
# the *runtime* package - step 2 below exposes it via a .pth file, and the H3
# kernels reach it by their own path. The CMake external project needs upstream
# vllm-flash-attn, which carries both the FA3 sources and the CUTLASS example
# headers that csrc/attention/sm70_v37/*.cu include. Aiming that variable at the
# Volta tree makes the build (1) request a target CMake never creates -
# "unknown target '_vllm_fa3_C'" - and (2) fail with
# "gemm_with_epilogue_visitor.h: No such file or directory" while compiling
# _vllm_fa2_C. Both were observed by building this script's output from a clean
# checkout.
# No VLLM_FLASH_ATTN_SM70 here on purpose: it is a CMake variable that the
# build derives from TORCH_CUDA_ARCH_LIST (CMakeLists.txt sets it when the
# 7.0 intersection is non-empty), and nothing reads it from the environment.
# TORCH_CUDA_ARCH_LIST above is the only selector that matters.

# Optional: seed these from local checkouts if your network to github is flaky.
# CUTLASS and triton-kernels are otherwise cloned during the build.
# export VLLM_CUTLASS_SRC_DIR=/path/to/cutlass
# export TRITON_KERNELS_SRC_DIR=/path/to/triton_kernels

echo "== venv =="
if [ ! -d "$VENV" ]; then
  python3.12 -m venv "$VENV"
fi
"$VENV/bin/pip" install --upgrade pip setuptools wheel cython packaging ninja \
  setuptools-rust setuptools-scm >/dev/null
"$VENV/bin/pip" install numpy pyyaml requests tqdm safetensors

# torch must be the cu128 build for CUDA 12.8 / driver 580.
if [ -n "${TORCH_WHL:-}" ]; then
  "$VENV/bin/pip" install "$TORCH_WHL"
fi

echo "== build (in-source, writes build.log) =="
# Both flags matter, and the second one was learned the hard way.
#
# --no-build-isolation: setuptools' `develop` now delegates to
#   `pip install -e . --use-pep517`, and an ISOLATED build installs pyproject's
#   build requirements into a scratch environment - which pins `torch == 2.13.0`
#   while this port runs the venv's own cu128 torch.
# --no-deps: without it, pip resolves and installs the project's RUNTIME
#   dependencies into YOUR venv, replacing the torch you installed with PyPI's
#   `torch 2.13.0`. That build wants the CUDA 13 runtime; on a driver-580 /
#   CUDA-12.8 host it fails to import at all with
#   `OSError: libcudart.so.13: cannot open shared object file`. It does this to a
#   working environment as a side effect of "building the extension".
#
# With both flags, the build uses the torch already in the venv - which is what
# the rest of this script installs and what the reference configuration runs.
( cd "$TREE" && "$VENV/bin/python" -m pip install -e . --no-build-isolation --no-deps 2>&1 | tee build.log )

# ---------------------------------------------------------------------------
# Two post-build steps that are easy to miss and produce confusing failures.
# ---------------------------------------------------------------------------

# 1. The build leaves _C_stable_libtorch.abi3.so in the REPO ROOT, but it is
#    imported from the vllm/ PACKAGE directory. Without this copy you get an
#    ImportError far from the cause.
if [ -f "$TREE/_C_stable_libtorch.abi3.so" ]; then
  # Copy to a temporary name and rename it into place. `cp -f` onto a .so that a
  # running worker has mmap'd keeps the SAME inode, so the live process faults
  # into a half-old, half-new file and dies inside whichever kernel was
  # executing - it is not a clean failure. A rename installs a new inode and
  # leaves any running process on the old one.
  _so_tmp="$TREE/vllm/.sm70-_C.$$"
  cp -f "$TREE/_C_stable_libtorch.abi3.so" "$_so_tmp"
  mv -f "$_so_tmp" "$TREE/vllm/_C_stable_libtorch.abi3.so"
  echo "installed _C_stable_libtorch.abi3.so into $TREE/vllm/ (atomic rename)"
fi

# 2. The two vendored extensions are not site-packages installs; expose them on
#    sys.path with .pth files. Imports happen per worker process, so this must
#    exist before the server starts.
SITE="$("$VENV/bin/python" -c 'import site;print(site.getsitepackages()[0])')"
if [ -d "$TREE/flash-attention-v100/flash_attn_v100" ]; then
  echo "$TREE/flash-attention-v100" > "$SITE/sm70-flash-attention-v100.pth"
  echo "wrote $SITE/sm70-flash-attention-v100.pth"
fi
if [ -d "$TREE/flash_qla" ]; then
  echo "$TREE" > "$SITE/sm70-flash-qla.pth"
  echo "wrote $SITE/sm70-flash-qla.pth"
fi

cat <<'EOF'

Build finished. Verify before serving:

  python -c "import flash_attn_v100, flash_qla; print('vendored extensions OK')"
  python -c "import vllm._C_stable_libtorch as m; print('stable ABI extension OK')"

Then check the server log for the SM70 kernels actually being selected:
  - "SM70 compressed-tensors NVFP4 TurboMind W4A16 dense path enabled"
  - "TURBOQUANT decode using Flash-V100 packed-cache path"   (only if the flag is on)

If the second line is absent, the decode kernel silently fell back to the
slower Triton path. See KNOWN-LIMITS.md section 2.
EOF
