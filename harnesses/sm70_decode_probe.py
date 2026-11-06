"""Isolate the Flash-V100 decode op from the engine.

Builds a fused (B, H, N, 2*D) KV cache exactly like this tree does, splits it
with the ported helper, and compares the extension's decode against a plain
torch reference. Answers one question only: is the layout/split the kernels get
correct, or is the garbage output coming from the attention math itself.
"""

import torch

from flash_attn_v100 import flash_attn_decode_paged
from vllm.v1.attention.backends.flash_attn_v100 import _split_paged_kv_cache

torch.manual_seed(0)
D = 256
H_kv = 2
BLOCK = 4
NBLK = 8
B = 1
SEQ = 32  # every block exactly full, so page bookkeeping cannot skew the ref
GQA = 2  # q heads per kv head

kv = torch.randn(NBLK, H_kv, BLOCK, 2 * D, device="cuda", dtype=torch.float16) * 0.5
k_cache, v_cache = _split_paged_kv_cache(kv)

q = torch.randn(B, H_kv * GQA, D, device="cuda", dtype=torch.float16) * 0.5
block_table = torch.tensor([list(range(NBLK))], device="cuda", dtype=torch.int32)
seq_lens = torch.tensor([SEQ], device="cuda", dtype=torch.int32)
out = torch.empty(B, H_kv * GQA, D, device="cuda", dtype=torch.float16)

res = flash_attn_decode_paged(
    q,
    k_cache,
    v_cache,
    block_table,
    seq_lens,
    softmax_scale=D**-0.5,
    out=out,
    kv_cache_dtype="auto",
)

# Reference: gather the same pages, no paging.
pages = block_table[0].tolist()
# Pages are addressed page-major: index (page, block_position, head, dim).
# Stacking per-page [H, BLOCK, D] slices and reshaping would interleave heads
# with block positions, which is a different sequence order.
# Reference rows are the *tensors actually passed in*, page-major. Deriving
# them from `kv` a second time would only re-test the split helper.
k_all = k_cache.contiguous().to(torch.float32)  # (NBLK, BLOCK, H, D)
v_all = v_cache.contiguous().to(torch.float32)
k_pages = k_all[pages].reshape(-1, H_kv, D)[:SEQ]
v_pages = v_all[pages].reshape(-1, H_kv, D)[:SEQ]

qf = q[0].float().reshape(H_kv, GQA, D)
ref = torch.zeros(H_kv * GQA, D, device="cuda", dtype=torch.float32)
for h in range(H_kv):
    logits = torch.einsum("gd,sd->sg", qf[h], k_pages[:, h]) * (D**-0.5)
    probs = torch.softmax(logits, dim=-1)
    ref[h * GQA : (h + 1) * GQA] = torch.einsum("sg,sd->gd", probs, v_pages[:, h])

got = res.float()[0]
err = (got - ref).abs()
print("out shape      :", tuple(got.shape))
print("max abs err    :", err.max().item())
print("mean abs err   :", err.mean().item())
print("ref magnitude  :", ref.abs().mean().item())
print("VERDICT        :", "kernel+layout OK" if err.max().item() < 5e-2 else "MISMATCH")

# Same check through a fully contiguous split (no fused row stride), to separate
# "stride handling" from "kernel semantics".
k_c = k_cache.contiguous()
v_c = v_cache.contiguous()
res2 = flash_attn_decode_paged(
    q, k_c, v_c, block_table, seq_lens, softmax_scale=D**-0.5, out=out, kv_cache_dtype="auto"
)
err2 = (res2.float()[0] - ref).abs()
print("contiguous copy max abs err:", err2.max().item())
