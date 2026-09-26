#!/usr/bin/env python3
"""纯反向基准（不含 forward）：FA2.7.4 / FA3.0.0 / TE2.14，覆盖定长 + 变长（varlen）。

定长：FA/TE 用 [B,S,H,D]；FA3 用同一接口。
变长：packed [T,H,D] + cu_seqlens。FA2.7.4 的 `flash_attn_varlen_func` **反向可用**
      （与 ROADMAP 旧记「FA2 反向不支持 varlen」相反，本轮实测更正）；FA3 变长反向可用；
      TE2.14 反向的 ragged QKV 在本容器报错/非法访存，故 varlen 只给 FA2/FA3 两列。
FA：fwd 建图一次（retain_graph），计时只跑 autograd.grad。
TE：fwd 一次拿 aux_ctx，计时只跑 fused_attn_bwd。
口径：CUPTI 纯 device 时间（所有 kernel 之和 / 次数），与 te-perf 的 fused_attn_bwd 一致。
      FLOPs（bwd）≈ 4·b·s²·h·(qk_dim+v_dim)=4·B·S²·H·2D（causal 未减半，与既有口径一致）；
      varlen 用 4·H·D·Σ_b L_b²。

用法：python fa_vs_te_bwd_only.py [fp16|bf16] [--verify]
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
    (1, 512, 16, 128, 16, True, "MHA S512"),
    (1, 256, 2, 512, 2, True, "MLA d512"),
    (1, 512, 4, 512, 4, True, "MLA d512"),
    (1, 1024, 2, 512, 2, True, "MLA d512"),
    (1, 4096, 16, 128, 16, True, "MHA S4096"),
]

# 变长（packed [T,H,D] + cu_seqlens）：(lengths, H, D, Hkv, causal, 说明)
# 覆盖与 harness/fa_bwd_bench.py 的 VARLEN_SHAPES 对应的生产形状（D≤128 故 FA2/FA3 均可用）。
VARLEN_SHAPES = [
    ([512, 1024, 2048, 256], 16, 128, 16, True, "MHA 不齐"),
    ([1024, 1024, 1024, 1024], 16, 128, 16, True, "MHA 等长"),
    ([128, 256, 512, 1024, 2048], 32, 128, 8, True, "GQA q32/kv8"),
    ([2048, 512, 128, 96, 64, 32, 16, 8], 16, 128, 16, True, "强倾斜"),
    ([1024, 1024, 1024, 1024], 16, 128, 16, False, "MHA 等长 full"),
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


def _cu(lengths):
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=DEV)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32, device=DEV).cumsum(0)
    return cu


def bench_fa_varlen(lengths, H, D, Hkv, causal, dt):
    """FA2.7.4 变长反向（packed [T,H,D] + cu_seqlens，autograd）。"""
    from flash_attn import flash_attn_varlen_func
    T = sum(lengths)
    cu = _cu(lengths)
    smax = max(lengths)
    torch.manual_seed(0)
    q = torch.randn(T, H, D, device=DEV, dtype=dt).requires_grad_(True)
    k = torch.randn(T, Hkv, D, device=DEV, dtype=dt).requires_grad_(True)
    v = torch.randn(T, Hkv, D, device=DEV, dtype=dt).requires_grad_(True)
    do = torch.randn(T, H, D, device=DEV, dtype=dt)
    o = flash_attn_varlen_func(q, k, v, cu, cu, smax, smax, causal=causal)

    def f():
        torch.autograd.grad(o, [q, k, v], do, retain_graph=True)
    return dev_time(f)


def bench_fa3_varlen(lengths, H, D, Hkv, causal, dt):
    """FA3.0.0 变长反向（packed + cu_seqlens，autograd）。"""
    from flash_attn_3 import flash_attn_interface as f3
    T = sum(lengths)
    cu = _cu(lengths)
    smax = max(lengths)
    torch.manual_seed(0)
    q = torch.randn(T, H, D, device=DEV, dtype=dt).requires_grad_(True)
    k = torch.randn(T, Hkv, D, device=DEV, dtype=dt).requires_grad_(True)
    v = torch.randn(T, Hkv, D, device=DEV, dtype=dt).requires_grad_(True)
    do = torch.randn(T, H, D, device=DEV, dtype=dt)
    o = f3.flash_attn_varlen_func(q, k, v, cu, cu, smax, smax, causal=causal)

    def f():
        torch.autograd.grad(o, [q, k, v], do, retain_graph=True)
    return dev_time(f)


def ref_varlen(q, k, v, do, lengths, causal=True):
    """fp32 autograd 参考（逐序列切片，用于校验 varlen 反向两列是否可信）。"""
    o, dq, dk, dv, off = [], [], [], [], 0
    for L in lengths:
        s = slice(off, off + L)
        qq = q[s].float().unsqueeze(0).detach().requires_grad_(True)
        kk = k[s].float().unsqueeze(0).detach().requires_grad_(True)
        vv = v[s].float().unsqueeze(0).detach().requires_grad_(True)
        qh, kh, vh = qq.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2)
        sc = (qh @ kh.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if causal:
            m = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), 1)
            sc = sc.masked_fill(m, float("-inf"))
        p = torch.softmax(sc, -1)
        oo = (p @ vh).transpose(1, 2)
        oo.backward(do[s].float().unsqueeze(0))
        o.append(oo.detach().squeeze(0)); dq.append(qq.grad.squeeze(0))
        dk.append(kk.grad.squeeze(0)); dv.append(vv.grad.squeeze(0))
        off += L
    return torch.cat(o), torch.cat(dq), torch.cat(dk), torch.cat(dv)


def _fa2_varlen(q, k, v, do, lengths, causal=True):
    from flash_attn import flash_attn_varlen_func
    cu, smax = _cu(lengths), max(lengths)
    q2, k2, v2 = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)
    o = flash_attn_varlen_func(q2, k2, v2, cu, cu, smax, smax, causal=causal)
    o.backward(do)
    return o.detach(), q2.grad, k2.grad, v2.grad


def _fa3_varlen(q, k, v, do, lengths, causal=True):
    from flash_attn_3 import flash_attn_interface as f3
    cu, smax = _cu(lengths), max(lengths)
    q2, k2, v2 = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)
    o = f3.flash_attn_varlen_func(q2, k2, v2, cu, cu, smax, smax, causal=causal)
    o.backward(do)
    return o.detach(), q2.grad, k2.grad, v2.grad


def verify_varlen(dt):
    """对一只小 shape 校验 FA2/FA3 变长反向 vs fp32 ref 的 max_abs（证明两列可信）。"""
    lengths = [128, 256, 64]
    H, D = 4, 128
    T = sum(lengths)
    torch.manual_seed(0)
    q = torch.randn(T, H, D, device=DEV, dtype=dt)
    k = torch.randn(T, H, D, device=DEV, dtype=dt)
    v = torch.randn(T, H, D, device=DEV, dtype=dt)
    do = torch.randn(T, H, D, device=DEV, dtype=dt)
    _, rdq, rdk, rdv = ref_varlen(q, k, v, do, lengths)
    line = f"varlen verify (causal) lengths={lengths} H{H} D{D} dt={str(dt)[6:]}: "
    for name, fn in (("fa2", _fa2_varlen), ("fa3", _fa3_varlen)):
        try:
            _, dq, dk, dv = fn(q, k, v, do, lengths)
            md = [(dq.float() - rdq).abs().max().item(),
                  (dk.float() - rdk).abs().max().item(),
                  (dv.float() - rdv).abs().max().item()]
            line += f"{name} dq/dk/dv={md[0]:.2e}/{md[1]:.2e}/{md[2]:.2e}  "
        except Exception as e:  # noqa
            line += f"{name}=FAIL({str(e)[:40]})  "
    print(line)


def main():
    args = [a for a in sys.argv[1:] if a != "--verify"]
    do_verify = "--verify" in sys.argv
    dt = DT[args[0] if args else "fp16"]

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

    print()
    print(f"pure bwd VARLEN (packed [T,H,D] + cu_seqlens; flops=4*H*D*sum(L^2)), dtype={dt}")
    print(f"{'lengths / H,kv / mask':30s} {'FA2.7.4':>15s} {'FA3':>15s} {'TE2.14':>15s} {'FA3/FA2':>8s}")
    for (lengths, H, D, Hkv, causal, label) in VARLEN_SHAPES:
        flops = 4.0 * H * D * sum(L * L for L in lengths)
        tag = f"{label} {lengths[:3]}{'..' if len(lengths) > 3 else ''} H{H}/kv{Hkv} {'causal' if causal else 'full'}"
        row = {}
        for who, fn in (("fa2", bench_fa_varlen), ("fa3", bench_fa3_varlen)):
            try:
                ms = fn(lengths, H, D, Hkv, causal, dt)
                row[who] = (ms, flops / (ms * 1e-3) / 1e12)
            except Exception as e:  # noqa
                row[who] = (float("nan"), float("nan"))
                print(f"  {who} failed {label}: {str(e)[:70]}")
        # TE2.14 反向的 ragged QKV 在本容器报错/非法访存（见文件头）；varlen 无 TE 列。
        f2 = row["fa2"]
        f3 = row["fa3"]
        r = f2[0] / f3[0] if f2[0] == f2[0] and f3[0] == f3[0] else float("nan")
        print(f"{tag:30s} "
              f"{f2[0]:6.4f}/{f2[1]:5.0f} {f3[0]:6.4f}/{f3[1]:5.0f} "
              f"{'NA':>15s} {r:7.2f}x")

    if do_verify:
        print()
        verify_varlen(dt)


if __name__ == "__main__":
    main()
