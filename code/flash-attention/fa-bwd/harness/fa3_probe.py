#!/usr/bin/env python3
"""FA3 (flash_attn_3) 可用性 + 数值对拍探针。"""
import math

import torch

import flash_attn_3
from flash_attn_3 import flash_attn_interface as f3

print("flash_attn_3", getattr(flash_attn_3, "__version__", "?"),
      "funcs:", [x for x in dir(f3) if "func" in x][:8])


def ref(q, k, v, do):
    D = q.shape[-1]
    S = q.shape[1]
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    kk = k.transpose(1, 2)
    vv = v.transpose(1, 2)
    if kk.shape[1] != q.shape[2]:
        rep = q.shape[2] // kk.shape[1]
        kk = kk.repeat_interleave(rep, dim=1)
        vv = vv.repeat_interleave(rep, dim=1)
    s = (q.transpose(1, 2) @ kk.transpose(-1, -2)) / math.sqrt(D)
    mask = torch.triu(torch.ones(S, S, device="cuda", dtype=torch.bool), 1)
    s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, -1)
    o = (p @ vv).transpose(1, 2)
    o.backward(do)
    return o.detach(), q.grad, k.grad, v.grad


for (B, S, H, D, Hkv) in [(1, 512, 16, 128, 16), (1, 1024, 40, 128, 8), (1, 1024, 64, 128, 1)]:
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
    do = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    o, dq, dk, dv = ref(q.float(), k.float(), v.float(), do.float())
    q2 = q.clone().requires_grad_(True)
    k2 = k.clone().requires_grad_(True)
    v2 = v.clone().requires_grad_(True)
    try:
        out = f3.flash_attn_func(q2, k2, v2, causal=True)
        out.backward(do)
        md = [max((a.float() - b.float()).abs().max().item() for a, b in [(out, o)]),
              (q2.grad.float() - dq.float()).abs().max().item(),
              (k2.grad.float() - dk.float()).abs().max().item(),
              (v2.grad.float() - dv.float()).abs().max().item()]
        print(f"B{B} S{S} H{H} D{D} kv{Hkv}: FA3 o/dq/dk/dv maxdiff = "
              f"{md[0]:.2e} / {md[1]:.2e} / {md[2]:.2e} / {md[3]:.2e}")
    except Exception as e:  # noqa
        print(f"B{B} S{S} H{H} D{D} kv{Hkv}: FA3 failed: {repr(e)[:200]}")
