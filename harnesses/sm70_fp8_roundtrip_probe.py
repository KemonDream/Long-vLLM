#!/usr/bin/env python
"""Does the Triton FP8 dtype round-trip correctly on Volta?

turboquant_k8v4 stores keys as FP8. The store kernel picks the FP8 dialect with

    _use_fp8_e4b15(): 1 if compute_capability < (8, 9) else 0

which was written for Ampere/Ada but is also true on Volta (7.0). Volta has no
FP8 hardware, so the question is whether Triton's software path for
float8e4b15 / float8e4nv actually round-trips on SM70.

Expected RMS relative error for an FP8 e4m3 round-trip (3 mantissa bits) is
about 2-4%. Anything much larger means the cast is not usable, which would
explain an 8-bit-key KV preset measuring worse than a 4-bit one.
"""

import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _rt_e4b15(x_ptr, o_ptr, N, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = x.to(tl.float8e4b15)
    tl.store(o_ptr + i, y.to(tl.float32), mask=i < N)


@triton.jit
def _rt_e4nv(x_ptr, o_ptr, N, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = x.to(tl.float8e4nv)
    tl.store(o_ptr + i, y.to(tl.float32), mask=i < N)


def roundtrip(fn, x):
    o = torch.zeros_like(x)
    n = x.numel()
    fn[(1,)](x, o, n, BLOCK=triton.next_power_of_2(n))
    return o


def main():
    torch.manual_seed(0)
    cap = torch.cuda.get_device_capability()
    print(f"device: {torch.cuda.get_device_name()}  capability {cap}")
    # Keys are l2-normalised before FP8 storage, so the magnitudes of interest
    # are the coordinates of a unit vector in D=256: sigma = 1/sqrt(256).
    n = 65536
    x = (torch.randn(n, device="cuda", dtype=torch.float32) / 16.0)
    for name, fn in (("float8e4b15", _rt_e4b15), ("float8e4nv", _rt_e4nv)):
        try:
            y = roundtrip(fn, x)
            err = (y - x).abs()
            rel = (err.norm() / x.norm()).item()
            print(f"  {name:12s} rel_rms {rel*100:7.3f}%  max|e| {err.max().item():.6f} "
                  f"nan {int(torch.isnan(y).sum())}  zero_out {int((y == 0).sum())}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:12s} FAILED: {type(exc).__name__}: {exc}")
    # Reference: what an ideal 3-bit Lloyd-Max coordinate quantiser would cost,
    # so the FP8 numbers can be read against the MSE presets' error budget.
    print("\n  for scale: a b-bit uniform/Lloyd-Max coordinate quantiser gives")
    for b, d in ((3, 0.03454), (4, 0.009497)):
        print(f"    {b}-bit  rel_rms {d ** 0.5 * 100:.2f}%")


if __name__ == "__main__":
    sys.exit(main())
