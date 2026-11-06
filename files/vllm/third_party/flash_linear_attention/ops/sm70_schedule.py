# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Added for this patch set: pinned Triton kernel schedules for SM70 (Volta),
# which stop the autotuner from exhausting GPU memory during warmup. The
# surrounding flash_linear_attention tree is FLA (MIT, Songlin Yang /
# Yu Zhang) vendored by upstream vLLM; this file is an addition.

"""SM70 (V100) autotune-schedule pinning for the Triton FLA GDN prefill kernels.

The Triton chunked-GDN kernels ship a broad autotune space tuned for Hopper and
newer parts. On Volta that search is both expensive and noisy: the winner varies
between runs, and several candidates land on configs that serialise badly on 80
SMs. 1Cat-vLLM pins a single schedule per kernel (kept here behind the same env
names) so the accepted route does not depend on Triton autotune winner variance.

These pins only affect the Triton fallback route; when the fused SM70 FlashQLA
backend is active (``VLLM_SM70_GDN_FLASHQLA_PREFILL``) these kernels do not run at
all.

Override knobs are read directly from the environment on purpose - they exist for
diagnostics and are not part of the supported config surface.
"""

import os


def _sm70_capability() -> bool:
    """True on CC 7.0 (V100). Never initialises CUDA just to answer the question."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        return major == 7 and minor == 0
    except Exception:  # noqa: BLE001 - import-time probe must never raise
        return False


def sm70_schedule_enabled(env_name: str) -> bool:
    """Schedule pinning for this kernel is on and we are actually on SM70."""
    if os.getenv(env_name, "1") != "1":
        return False
    return _sm70_capability()


def sm70_ints(env_name: str, default: list[int]) -> list[int]:
    """Parse a comma/space separated positive-int list, else return ``default``."""
    raw = os.getenv(env_name)
    if not raw:
        return list(default)
    values: list[int] = []
    for token in raw.replace(",", " ").split():
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return values or list(default)
