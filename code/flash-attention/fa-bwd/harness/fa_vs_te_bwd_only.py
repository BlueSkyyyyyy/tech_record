#!/usr/bin/env python3
"""纯反向基准（不含 forward）：FA2.7.4 vs TE2.14，覆盖用户指定的 GQA/MQA/MLA 形状。

FA：fwd 建图一次（retain_graph），计时只跑 autograd.grad。
TE：fwd 一次拿 aux_ctx，计时只跑 fused_attn_bwd。
口径：CUPTI 纯 device 时间（所有 kernel 之和 / 次数），与 te-perf 的 fused_attn_bwd 一致。

用法：python fa_vs_te_bwd_only.py [fp16|bf16]
"""
import math
import sys

import torch
from torch.profiler import ProfilerActivity, profile

DEV = "cuda"
DT = {"fp16": torch.float16, "bf16": torch.bfloat16}
SHAPES = [  # (B,S,H,D,Hkv,causal, 说明)
    (1, 1024, 40, 128, 8, True, "GQA q40/kv8"),
    (1, 1024, 32, 128, 4, True, "GQA q32/kv4"),
    (1, 1024, 64, 128, 4, True, "GQA q64/kv4"),
    (1, 1024, 64, 128, 1, True, "MQA q64/kv1"),
    (1, 256, 2, 512, 2, True, "MLA d512"),
    (1, 512, 4, 512, 4, True, "MLA d512"),
    (1, 1024, 2, 512, 2, True, "MLA d512"),
    (1, 4096, 16, 128, 16, True, "MHA S4096"),
]


def dev_time(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in prof.key_averages()) / repeat / 1e3


def bench_fa(B, S, H, D, Hkv, causal, dt):
    from flash_attn import flash_attn_func
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device=DEV, dtype=dt).requires_grad_(True)
    k = torch.randn(B, S, Hkv, D, device=DEV, dtype=dt).requires_grad_(True)
    v = torch.randn(B, S, Hkv, D, device=DEV, dtype=dt).requires_grad_(True)
    do = torch.randn(B, S, H, D, device=DEV, dtype=dt)
    o = flash_attn_func(q, k, v, causal=causal)

    def f():
        torch.autograd.grad(o, [q, k, v], do, retain_graph=True)
    return dev_time(f)


def bench_fa3(B, S, H, D, Hkv, causal, dt):
    from flash_attn_3 import flash_attn_interface as f3
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device=DEV, dtype=dt).requires_grad_(True)
    k = torch.randn(B, S, Hkv, D, device=DEV, dtype=dt).requires_grad_(True)
    v = torch.randn(B, S, Hkv, D, device=DEV, dtype=dt).requires_grad_(True)
    do = torch.randn(B, S, H, D, device=DEV, dtype=dt)
    o = f3.flash_attn_func(q, k, v, causal=causal)

    def f():
        torch.autograd.grad(o, [q, k, v], do, retain_graph=True)
    return dev_time(f)


def bench_te(B, S, H, D, Hkv, causal, dt):
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        FusedAttnBackend, fused_attn_bwd, fused_attn_fwd,
    )
    from transformer_engine.pytorch.constants import TE_DType
    torch.manual_seed(0)
    q = torch.randn(B * S, H, D, device=DEV, dtype=dt)
    k = torch.randn(B * S, Hkv, D, device=DEV, dtype=dt)
    v = torch.randn(B * S, Hkv, D, device=DEV, dtype=dt)
    do = torch.randn(B * S, H, D, device=DEV, dtype=dt)
    cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=DEV)
    mask = "causal" if causal else "no_mask"
    backend = FusedAttnBackend["F16_arbitrary_seqlen"]
    out, aux = fused_attn_fwd(True, S, S, cu, cu, q, k, v, dt, backend, None,
                              attn_bias_type="no_bias", attn_mask_type=mask,
                              softmax_type="vanilla", qkv_layout="bshd_bshd_bshd")

    def f():
        fused_attn_bwd(S, S, cu, cu, q, k, v, out, do, dt,
                       qkv_layout="bshd_bshd_bshd", dqkv_dtype=TE_DType[dt],
                       aux_ctx_tensors=list(aux), fused_attention_backend=backend,
                       attn_bias_type="no_bias", attn_mask_type=mask, softmax_type="vanilla")
    return dev_time(f)


def main():
    dt = DT[sys.argv[1] if len(sys.argv) > 1 else "fp16"]
    print(f"pure bwd device time (ms / TFLOPS @4BS^2H(D+Dv)), dtype={dt}")
    print(f"{'shape':30s} {'FA2.7.4':>15s} {'FA3':>15s} {'TE2.14':>15s} {'FA3/FA2':>8s}")
    for (B, S, H, D, Hkv, causal, label) in SHAPES:
        flops = 4.0 * B * S * H * S * (2 * D)
        row = {}
        for who, fn in (("fa2", bench_fa), ("fa3", bench_fa3), ("te", bench_te)):
            try:
                ms = fn(B, S, H, D, Hkv, causal, dt)
                row[who] = (ms, flops / (ms * 1e-3) / 1e12)
            except Exception as e:  # noqa
                row[who] = (float("nan"), float("nan"))
                print(f"  {who} failed {label}: {str(e)[:70]}")
        f2, f3, te = row["fa2"], row["fa3"], row["te"]
        r = f2[0] / f3[0] if f2[0] == f2[0] and f3[0] == f3[0] else float("nan")
        print(f"{(str((B,S,H,D))+' kv='+str(Hkv)):30s} "
              f"{f2[0]:6.4f}/{f2[1]:5.0f} {f3[0]:6.4f}/{f3[1]:5.0f} {te[0]:6.4f}/{te[1]:5.0f} {r:7.2f}x")


if __name__ == "__main__":
    main()
