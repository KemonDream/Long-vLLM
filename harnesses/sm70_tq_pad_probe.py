#!/usr/bin/env python3
"""Prove that front-padding a short continuation chunk to 1024 rows is exactly
equivalent to the dense SDPA continuation path.

The SM70 FA2 kernel rejects q_len < 1024, and dense SDPA over a long prefix OOMs the
engine, so short continuation chunks have to go through FA2 padded up. That is only
sound if the padded call reproduces the same causal alignment: query r (0-based within
the chunk) must attend to keys 0..cached_len+r, i.e. bottom-right alignment.

Reference = the dense path's own mask (k_pos <= q_pos, q_pos = arange(q_len)+cached_len).
Candidate = FA2 varlen on q front-padded to 1024 with cu_seqlens_q=(0,1024), sliced back.
"""
import os
import torch

torch.manual_seed(0)
dev = torch.device("cuda:0")

N, Hq, Hk, D = int(os.environ.get("P_N", "300")), 20, 2, 256
SEQ = int(os.environ.get("P_SEQ", "5000"))
MIN_Q = 1024
cached_len = SEQ - N

q = (torch.randn(N, Hq, D, device=dev, dtype=torch.float16) * 0.1)
k = (torch.randn(SEQ, Hk, D, device=dev, dtype=torch.float16) * 0.1)
v = (torch.randn(SEQ, Hk, D, device=dev, dtype=torch.float16) * 0.1)
scale = D ** -0.5

# ---- reference: dense SDPA exactly as _continuation_prefill does it ----
q_t = q.transpose(0, 1).unsqueeze(0)
k_t = k.transpose(0, 1).unsqueeze(0)
v_t = v.transpose(0, 1).unsqueeze(0)
q_pos = torch.arange(N, device=dev).unsqueeze(1) + cached_len
k_pos = torch.arange(SEQ, device=dev).unsqueeze(0)
mask = k_pos <= q_pos
ref = torch.nn.functional.scaled_dot_product_attention(
    q_t, k_t, v_t, attn_mask=mask, scale=scale, enable_gqa=(Hk < Hq)
)[0].transpose(0, 1).float()

# ---- candidate: FA2 with front padding ----
from vllm.vllm_flash_attn import flash_attn_varlen_func

pad = MIN_Q - N
qp = torch.empty(MIN_Q, Hq, D, device=dev, dtype=torch.float16)
qp[pad:] = q
cu_q = torch.tensor([0, MIN_Q], device=dev, dtype=torch.int32)
cu_k = torch.tensor([0, SEQ], device=dev, dtype=torch.int32)
out = flash_attn_varlen_func(
    q=qp, k=k, v=v,
    cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
    max_seqlen_q=MIN_Q, max_seqlen_k=SEQ,
    causal=True, softmax_scale=scale,
)
cand = out[pad:].float()

d = (ref - cand).abs()
rel = d.max().item() / (ref.abs().max().item() + 1e-9)
print(f"q_len={N} padded->{MIN_Q}  seq_len={SEQ}  prefix={cached_len}")
print(f"max|diff| = {d.max().item():.3e}   mean|diff| = {d.mean().item():.3e}   rel = {rel:.3e}")
print("PADDING EQUIVALENT" if rel < 5e-3 else "PADDING WRONG")

# Also verify the unpadded-but-legal case still matches (sanity on the ref path).
q2 = q.repeat(4, 1, 1)          # 1200 rows >= 1024
N2 = q2.shape[0]
cached2 = SEQ - N2              # keep seq_len == cached_len + q_len
assert N2 + cached2 == SEQ
k2 = k[: SEQ]
v2 = v[: SEQ]
ref2 = torch.nn.functional.scaled_dot_product_attention(
    q2.transpose(0, 1).unsqueeze(0), k2.transpose(0, 1).unsqueeze(0),
    v2.transpose(0, 1).unsqueeze(0),
    attn_mask=(torch.arange(k2.shape[0], device=dev).unsqueeze(0)
               <= torch.arange(N2, device=dev).unsqueeze(1) + cached2),
    scale=scale, enable_gqa=(Hk < Hq),
)[0].transpose(0, 1).float()
cand2 = flash_attn_varlen_func(
    q=q2, k=k2, v=v2,
    cu_seqlens_q=torch.tensor([0, N2], device=dev, dtype=torch.int32),
    cu_seqlens_k=torch.tensor([0, k2.shape[0]], device=dev, dtype=torch.int32),
    max_seqlen_q=N2, max_seqlen_k=k2.shape[0],
    causal=True, softmax_scale=scale,
).float()
rel2 = (ref2 - cand2).abs().max().item() / (ref2.abs().max().item() + 1e-9)
print(f"unpadded q_len={N2}: rel = {rel2:.3e}  ->", "OK" if rel2 < 5e-3 else "MISMATCH")
