#!/usr/bin/env python3
"""列出 FA 反向 vs TE 反向各自 launch 的 kernel 及 device 时间，用于归因性能差异。

用法（容器内）：
  python fa_vs_te_profile.py "1 1024 40 128 kv=8 causal" fp16
  python fa_vs_te_profile.py "1 4096 16 128 causal" fp16
"""
import math
import sys

import torch
from torch.profiler import ProfilerActivity, profile

DEV = "cuda"
DT = {"fp16": torch.float16, "bf16": torch.bfloat16}


def build(sh):
    B, S, H, D, Hkv = sh["B"], sh["S"], sh["H"], sh["D"], sh["Hkv"]
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device=DEV, dtype=sh["dt"])
    k = torch.randn(B, S, Hkv, D, device=DEV, dtype=sh["dt"])
    v = torch.randn(B, S, Hkv, D, device=DEV, dtype=sh["dt"])
    do = torch.randn(B, S, H, D, device=DEV, dtype=sh["dt"])
    return q, k, v, do


def fa_run(q, k, v, do, causal):
    from flash_attn import flash_attn_func
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    o = flash_attn_func(q2, k2, v2, causal=causal)
    o.backward(do)


def fa3_run(q, k, v, do, causal):
    from flash_attn_3 import flash_attn_interface as f3
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    o = f3.flash_attn_func(q2, k2, v2, causal=causal)
    o.backward(do)


def te_run(q, k, v, do, causal):
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        FusedAttnBackend, fused_attn_bwd, fused_attn_fwd,
    )
    from transformer_engine.pytorch.constants import TE_DType
    B, S, H, D = q.shape
    Hkv = k.shape[2]
    cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=DEV)
    dt = q.dtype
    qf = q.reshape(B * S, H, D).contiguous()
    kf = k.reshape(B * S, Hkv, D).contiguous()
    vf = v.reshape(B * S, Hkv, D).contiguous()
    mask = "causal" if causal else "no_mask"
    backend = FusedAttnBackend["F16_arbitrary_seqlen"]
    out, aux = fused_attn_fwd(True, S, S, cu, cu, qf, kf, vf, dt, backend, None,
                              attn_bias_type="no_bias", attn_mask_type=mask,
                              softmax_type="vanilla", qkv_layout="bshd_bshd_bshd")
    dof = do.reshape(B * S, H, D).contiguous()
    fused_attn_bwd(S, S, cu, cu, qf, kf, vf, out, dof, dt,
                   qkv_layout="bshd_bshd_bshd", dqkv_dtype=TE_DType[dt],
                   aux_ctx_tensors=list(aux), fused_attention_backend=backend,
                   attn_bias_type="no_bias", attn_mask_type=mask, softmax_type="vanilla")


def profile_kernels(fn, label):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
    evs = [(e.key, e.device_time_total / 10.0) for e in prof.key_averages()
           if e.device_time_total > 0]
    evs.sort(key=lambda x: -x[1])
    tot = sum(t for _, t in evs)
    print(f"\n===== {label}: total {tot/1e3:.4f} ms/call, {len(evs)} kernels =====")
    for name, us in evs[:15]:
        print(f"  {us:9.2f} us  {name[:100]}")


def main():
    spec = sys.argv[1].split()
    dt = DT[sys.argv[2] if len(sys.argv) > 2 else "fp16"]
    B, S, H, D = (int(x) for x in spec[:4])
    Hkv, causal = H, True
    for t in spec[4:]:
        if t.startswith("kv="):
            Hkv = int(t[3:])
        if t == "full":
            causal = False
    sh = dict(B=B, S=S, H=H, D=D, Hkv=Hkv, dt=dt)
    print(f"shape={sh} causal={causal}")
    q, k, v, do = build(sh)
    profile_kernels(lambda: fa_run(q, k, v, do, causal), "FA 2.7.4 bwd (SM80)")
    profile_kernels(lambda: fa3_run(q, k, v, do, causal), "FA3 bwd (SM90)")
    profile_kernels(lambda: te_run(q, k, v, do, causal), "TE 2.14 bwd")


if __name__ == "__main__":
    main()
