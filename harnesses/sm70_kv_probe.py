"""End-to-end op-level replica of the engine's SM70 KV path.

Reproduces exactly what FlashAttnV100Impl does at runtime (no engine, no
torch.compile):
  1. allocate the fused (B, H, N, 2*head_size) cache this tree creates,
  2. split it the way the write path does,
  3. write real K/V with the Triton reshape-and-cache kernel,
  4. read it back through flash_attn_decode_paged / flash_attn_prefill_paged,
  5. compare against plain torch attention on the same K/V.

If this matches, the kernels are fine and the engine garbage must come from
metadata (slot_mapping / block_table) or from a route other than the ones
exercised here.
"""

import torch

from flash_attn_v100 import flash_attn_decode_paged, flash_attn_prefill_paged
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)

torch.manual_seed(0)
D = 256
H_kv = 4
H_q = 4
BLOCK = 4
S = 16
NBLK = (S + BLOCK - 1) // BLOCK
DEV = "cuda"
DT = torch.float16

# 1. engine-style fused cache: logical (B, H, N, 2*D)
kv_cache = torch.zeros(NBLK, H_kv, BLOCK, 2 * D, device=DEV, dtype=DT)

# 2. engine-style split (write path uses self.head_size; here 2*D//2 == D)
k_cache, v_cache = kv_cache.transpose(1, 2).split(D, dim=-1)

key = torch.randn(S, H_kv, D, device=DEV, dtype=DT) * 0.5
value = torch.randn(S, H_kv, D, device=DEV, dtype=DT) * 0.5
slot_mapping = torch.arange(S, device=DEV, dtype=torch.int64)
k_scale = torch.tensor(1.0, device=DEV, dtype=torch.float32)
v_scale = torch.tensor(1.0, device=DEV, dtype=torch.float32)

# 3. write
triton_reshape_and_cache_flash(
    key, value, k_cache, v_cache, slot_mapping, "auto", k_scale, v_scale
)
torch.cuda.synchronize()

written_k = k_cache.reshape(-1, H_kv, D)[:S].float()
print("write ok      :", torch.allclose(written_k, key.float(), atol=1e-2))
print("write max err :", (written_k - key.float()).abs().max().item())

block_table = torch.arange(NBLK, device=DEV, dtype=torch.int32).unsqueeze(0)
seq_lens = torch.tensor([S], device=DEV, dtype=torch.int32)


def reference(q, k, v):
    logits = torch.einsum("hd,shd->hs", q.float(), k.float()) * (D**-0.5)
    probs = torch.softmax(logits, dim=-1)
    return torch.einsum("hs,shd->hd", probs, v.float())


# 4a. decode (q len 1, attends to all S)
q_dec = torch.randn(1, H_q, D, device=DEV, dtype=DT) * 0.5
out = torch.empty(1, H_q, D, device=DEV, dtype=DT)
res = flash_attn_decode_paged(
    q_dec, k_cache, v_cache, block_table, seq_lens,
    softmax_scale=D**-0.5, out=out, kv_cache_dtype="auto",
)
print("decode max err:", (res.float()[0] - reference(q_dec[0], key, value)).abs().max().item())

# 4b. prefill (q len S, causal)
q_pre = torch.randn(S, H_q, D, device=DEV, dtype=DT) * 0.5
out_pre = torch.empty(S, H_q, D, device=DEV, dtype=DT)
# prefill_paged wants (B, S_q, H, D); a single request keeps B=1.
res_pre = flash_attn_prefill_paged(
    q_pre.unsqueeze(0), k_cache, v_cache, block_table, seq_lens,
    softmax_scale=D**-0.5, out=out_pre.unsqueeze(0), kv_cache_dtype="auto",
    causal=True,
).squeeze(0)
ref_pre = torch.zeros(S, H_q, D, device=DEV, dtype=torch.float32)
for i in range(S):
    ref_pre[i] = reference(q_pre[i], key[: i + 1], value[: i + 1])
print("prefill max err:", (res_pre.float() - ref_pre).abs().max().item())

# 4c. XQA decode — the default route for fp16 KV in the engine.
from flash_attn_v100 import (
    flash_attn_decode_paged_xqa,
    flash_attn_decode_paged_xqa_available,
)

if flash_attn_decode_paged_xqa_available():
    out_xqa = torch.empty(1, H_q, D, device=DEV, dtype=DT)
    res_xqa = flash_attn_decode_paged_xqa(
        q_dec, k_cache, v_cache, block_table, seq_lens,
        softmax_scale=D**-0.5, out=out_xqa, kv_cache_dtype="auto",
    )
    print(
        "xqa decode err:",
        (res_xqa.float()[0] - reference(q_dec[0], key, value)).abs().max().item(),
    )
else:
    print("xqa decode err: unavailable")
