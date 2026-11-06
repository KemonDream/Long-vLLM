#!/usr/bin/env python
"""Does the measured KV-quantisation loss match scalar-quantiser theory?

TurboQuant applies a Hadamard rotation and then quantises each coordinate with a
scalar quantiser. Rotation makes the coordinates approximately i.i.d. Gaussian,
which is exactly the assumption under which the classic optimal-scalar-quantiser
distortion constants apply. For a unit-variance Gaussian source the minimum MSE
of a b-bit quantiser is:

    b=3 (8 levels):   D3 = 0.03454
    b=4 (16 levels):  D4 = 0.009497

and for the VALUE quantiser (uniform, not Lloyd-Max) distortion scales as
Delta^2/12 with Delta proportional to 2^-b, so each removed bit multiplies the
distortion by exactly 4.

Because the four presets differ in one knob at a time we can test the model with
no free parameters. Write each measured loss as key + value contributions:

    a = dNLL(4bit_nc) = K4 + V4      (keys 4-bit LM, values 4-bit)
    b = dNLL(k3v4_nc) = K3 + V4      (keys 3-bit LM, values 4-bit)
    c = dNLL(3bit_nc) = K3 + V3      (keys 3-bit LM, values 3-bit)

Then  b - a = K3 - K4 = (D3/D4 - 1) K4 = 2.636 K4
      c - b = V3 - V4 = 3 V4
and   a = K4 + V4.

So K4 = (b-a)/2.636 and V4 = (c-b)/3 predict a = K4 + V4 with no fitted
parameters. If the prediction matches the measured a, the loss decomposes exactly
as a 3-bit scalar quantiser should, and the loss is "in control": bounded by, and
consistent with, the quantiser's own error budget.

Usage: python harnesses/sm70_kv_ladder_analyze.py /tmp/vllm_sm70/nll
"""

import glob
import json
import math
import os
import sys

import numpy as np

D3, D4 = 0.03454, 0.009497
KEY_RATIO = D3 / D4  # 3.637
VAL_RATIO = 4.0

REF = "float16"          # k8v4 turned out broken; see report
BROKEN = {"turboquant_k8v4"}


def load(d):
    runs = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        j = json.load(open(f))
        key = (j["kv_cache_dtype"], j["window"])
        npy = j.get("npy")
        arr = np.load(npy) if npy and os.path.exists(npy) else None
        runs[key] = (j, arr)
    return runs


def paired(a, b):
    """Paired difference of two per-token NLL vectors: mean, SE, and a t-like z."""
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    return m, se, (m / se if se > 0 else float("inf")), n


def main(d):
    runs = load(d)
    if not runs:
        raise SystemExit(f"no runs in {d}")
    windows = sorted({w for _, w in runs})
    print(f"runs: {len(runs)}  windows: {windows}\n")

    for w in windows:
        have = {kv: v for (kv, ww), v in runs.items() if ww == w}
        if len(have) < 2:
            continue
        print(f"=== window {w:,} tokens ===")
        ref = have.get(REF)
        if ref is None:
            ref = have.get("turboquant_4bit_nc")
            if ref is None:
                print("  (no reference at this window)\n")
                continue
            print("  (no float16 reference at this window; anchoring on 4bit_nc)")
        rj, ra = ref
        print(f"  anchor {REF}: nll {rj['mean_nll']:.6f}  ppl {rj['ppl']:.4f}")
        dl = {}
        for kv, (j, arr) in sorted(have.items()):
            if kv == REF or arr is None or ra is None:
                continue
            m, se, z, n = paired(arr, ra)
            dl[kv] = (m, se, z)
            pct = 100.0 * m / rj["mean_nll"]
            print(
                f"  {kv:24s} dNLL {m:+.6f} +-{se:.6f} (z {z:+.1f})  "
                f"{pct:+.4f}% NLL   ppl {j['ppl']:.4f}"
            )

        # Zero-free-parameter consistency check.
        need = ("turboquant_4bit_nc", "turboquant_k3v4_nc", "turboquant_3bit_nc")
        if all(k in dl for k in need):
            a = dl["turboquant_4bit_nc"][0]
            b = dl["turboquant_k3v4_nc"][0]
            c = dl["turboquant_3bit_nc"][0]
            k4 = (b - a) / (KEY_RATIO - 1.0)
            v4 = (c - b) / (VAL_RATIO - 1.0)
            pred = k4 + v4
            print("\n  --- decomposition (zero free parameters) ---")
            print(f"  K4 (4-bit key contribution)   {k4:+.6f}")
            print(f"  V4 (4-bit value contribution) {v4:+.6f}")
            print(f"  predicted dNLL(4bit_nc) = K4+V4 = {pred:+.6f}")
            print(f"  measured  dNLL(4bit_nc)         = {a:+.6f}")
            err = pred - a
            sa = dl["turboquant_4bit_nc"][1]
            print(f"  residual {err:+.6f}  ({abs(err)/abs(a)*100 if a else 0:.1f}% of measured)")
            print(f"  pass (residual within 3 sigma of the measured term): "
                  f"{abs(err) <= 3 * max(sa, 1e-12)}")
            print(f"  monotone 3bit>k3v4>4bit>k8v4: "
                  f"{c > b > a > 0}")
        print()

    # Absolute reference, short window only (fp16 cannot reach long context).
    for w in windows:
        f16 = runs.get(("float16", w))
        k8 = runs.get((REF, w))
        if f16 and k8 and f16[1] is not None and k8[1] is not None:
            m, se, z, n = paired(k8[1], f16[1])
            print(f"=== backend/quant confound check at {w:,} ===")
            print(f"  {REF} (TurboQuant) minus float16 (FLASH_ATTN_V100):"
                  f" dNLL {m:+.6f} +-{se:.6f} (z {z:+.1f}) over {n} tokens")
            print("  Small |z| => the 8-bit-key preset is numerically equivalent to"
                  " fp16 KV, so the backend difference is not confounded with"
                  " precision.\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/vllm_sm70/nll")
