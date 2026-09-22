#!/usr/bin/env python3
"""FA 反向：参考值 dump + 数值对拍 + 性能基准（FA / TE 对照）。

用法（在 kernel_lab 容器内）：
  python fa_bwd_bench.py dump   [--shapes ...]   # 生成输入并 dump ref/fa/te 的 I/O 到 /home/xieminglin/proj/output/fa-bwd
  python fa_bwd_bench.py bench  [--dtype fp16]   # 用 CUPTI 测 FA / TE 反向的纯 device 时间
  python fa_bwd_bench.py all

设计要点：
- **输入用固定 seed 生成**，输入张量以 fp32 CPU npy 保存（fp16/bf16 可被 fp32 无损表示），
  方便自己的 kernel 直接 load 同一份输入、和同一份 ref 输出逐元素比对。
- ref = 纯 PyTorch fp32（autograd）；fa = flash_attn 2.7.4；te = TransformerEngine 2.14。
- 输出目录：/home/xieminglin/proj/output/fa-bwd/<case>/ ；meta.json 记录 shape/dtype/causal/seed。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

OUT_ROOT = Path("/home/xieminglin/proj/output/fa-bwd")
DEV = "cuda"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

# 默认测试形状：(B, S, H, D, causal)
DEFAULT_SHAPES = [
    (1, 512, 16, 128, True),
    (1, 1024, 16, 128, True),
    (4, 2048, 16, 128, True),
    (2, 2048, 16, 128, False),
    (1, 4096, 16, 128, True),
    (1, 1024, 32, 128, True),
    (1, 4096, 40, 128, True),  # Qwen3-8B GQA 用会另测；此处先当 MHA
]


def case_slug(B, S, H, D, causal, dtype):
    return f"b{B}_s{S}_h{H}_d{D}_{'causal' if causal else 'full'}_{dtype}"


def seed_case(B, S, H, D):
    torch.manual_seed(1234 + B * 31 + S * 7 + H * 3 + D)


# ----------------------------- 参考实现 -----------------------------
def ref_attn(q, k, v, do, causal=True):
    """fp32 autograd 参考。q/k/v/do: [B,S,H,D]; 返回 o,dq,dk,dv（detach）。"""
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    B, S, H, D = q.shape
    qq, kk, vv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    scale = 1.0 / math.sqrt(D)
    s = (qq @ kk.transpose(-1, -2)) * scale
    if causal:
        mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = (p @ vv).transpose(1, 2)
    o.backward(do)
    return o.detach(), q.grad, k.grad, v.grad


def fa_bwd(q, k, v, do, causal=True):
    from flash_attn import flash_attn_func
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    o = flash_attn_func(q2, k2, v2, causal=causal)
    o.backward(do)
    return o.detach(), q2.grad, k2.grad, v2.grad


def te_bwd(q, k, v, do, causal=True):
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        FusedAttnBackend, fused_attn_bwd, fused_attn_fwd,
    )
    from transformer_engine.pytorch.constants import TE_DType
    B, S, H, D = q.shape
    cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=DEV)
    backend = FusedAttnBackend["F16_arbitrary_seqlen"]
    dtype = q.dtype
    mask = "causal" if causal else "no_mask"
    qf, kf, vf = (x.reshape(B * S, H, D).contiguous() for x in (q, k, v))
    out, aux = fused_attn_fwd(
        True, S, S, cu, cu, qf, kf, vf, dtype, backend, None,
        attn_bias_type="no_bias", attn_mask_type=mask,
        softmax_type="vanilla", qkv_layout="bshd_bshd_bshd",
    )
    dof = do.reshape(B * S, H, D).contiguous()
    dqkv = fused_attn_bwd(
        S, S, cu, cu, qf, kf, vf, out, dof, dtype,
        qkv_layout="bshd_bshd_bshd", dqkv_dtype=TE_DType[dtype],
        aux_ctx_tensors=list(aux), fused_attention_backend=backend,
        attn_bias_type="no_bias", attn_mask_type=mask, softmax_type="vanilla",
    )
    o = out.reshape(B, S, H, D)
    return (o.detach(), dqkv[0].reshape(B, S, H, D), dqkv[1].reshape(B, S, H, D),
            dqkv[2].reshape(B, S, H, D))


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


# ----------------------------- dump -----------------------------
def dump_case(B, S, H, D, causal, dtype_name):
    dtype = DTYPES[dtype_name]
    seed_case(B, S, H, D)
    q = torch.randn(B, S, H, D, device=DEV, dtype=dtype)
    k = torch.randn(B, S, H, D, device=DEV, dtype=dtype)
    v = torch.randn(B, S, H, D, device=DEV, dtype=dtype)
    do = torch.randn(B, S, H, D, device=DEV, dtype=dtype)

    o_ref, dq_ref, dk_ref, dv_ref = ref_attn(q.float(), k.float(), v.float(), do.float(), causal)
    res = {"ref": (o_ref, dq_ref, dk_ref, dv_ref)}
    try:
        res["fa"] = fa_bwd(q, k, v, do, causal)
    except Exception as e:  # noqa
        print(f"  [fa] failed: {e}")
    try:
        res["te"] = te_bwd(q, k, v, do, causal)
    except Exception as e:  # noqa
        print(f"  [te] failed: {e}")

    slug = case_slug(B, S, H, D, causal, dtype_name)
    d = OUT_ROOT / slug
    d.mkdir(parents=True, exist_ok=True)

    def save(name, t):
        np.save(d / f"{name}.npy", t.detach().float().cpu().numpy())

    # 输入（fp32 无损保存 fp16/bf16 值）
    for name, t in (("q", q), ("k", k), ("v", v), ("do", do)):
        save(name, t)
    for who, (o, dq, dk, dv) in res.items():
        for name, t in (("o", o), ("dq", dq), ("dk", dk), ("dv", dv)):
            save(f"{who}_{name}", t)
    meta = {
        "B": B, "S": S, "H": H, "D": D, "causal": causal, "dtype": dtype_name,
        "seed": 1234 + B * 31 + S * 7 + H * 3 + D, "scale": 1.0 / math.sqrt(D),
        "which": list(res.keys()),
        "cmd": f"fa_bwd_bench.py dump --shape {B} {S} {H} {D} {'causal' if causal else 'full'} --dtype {dtype_name}",
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # 数值对拍摘要
    lines = [f"[{slug}]"]
    for who in ("fa", "te"):
        if who in res:
            lines.append(
                f"  {who:3s} vs ref: o {maxdiff(res[who][0], o_ref):.2e} "
                f"dq {maxdiff(res[who][1], dq_ref):.2e} dk {maxdiff(res[who][2], dk_ref):.2e} "
                f"dv {maxdiff(res[who][3], dv_ref):.2e}")
    print("\n".join(lines))
    return slug


# ----------------------------- bench -----------------------------
class CudaTimer:
    """CUPTI 纯 device 时间（与 te-perf 口径一致）。"""

    def __init__(self, warmup=10, repeat=50):
        self.warmup, self.repeat = warmup, repeat

    def device_time(self, fn):
        from torch.profiler import ProfilerActivity, profile
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(self.repeat):
                fn()
            torch.cuda.synchronize()
        total_us = sum(e.device_time_total for e in prof.key_averages())
        return total_us / self.repeat / 1e3  # us -> ms


def bench_case(B, S, H, D, causal, dtype_name, timer):
    dtype = DTYPES[dtype_name]
    seed_case(B, S, H, D)
    q = torch.randn(B, S, H, D, device=DEV, dtype=dtype)
    k = torch.randn(B, S, H, D, device=DEV, dtype=dtype)
    v = torch.randn(B, S, H, D, device=DEV, dtype=dtype)
    do = torch.randn(B, S, H, D, device=DEV, dtype=dtype)
    flops = 4.0 * B * S * H * S * D  # bwd 约 2x fwd
    out = []
    for who, fn in (("fa", fa_bwd), ("te", te_bwd)):
        try:
            ms = timer.device_time(lambda: fn(q, k, v, do, causal))
            out.append((who, ms, flops / (ms * 1e-3) / 1e12))
        except Exception as e:  # noqa
            out.append((who, float("nan"), float("nan")))
            print(f"  [{who}] bench failed: {e}")
    tag = case_slug(B, S, H, D, causal, dtype_name)
    print(f"[{tag}] " + "  ".join(f"{w}={m:.4f}ms/{t:.2f}TF" for w, m, t in out))
    return out


def parse_shape(s):
    # "B S H D causal|full"（也接受 ["B","S","H","D","causal"] 列表）
    parts = s.split() if isinstance(s, str) else list(s)
    return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]),
            (len(parts) < 5 or parts[4] == "causal"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "bench", "all"])
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--shape", nargs="+", action="append", default=None,
                    help="B S H D [causal|full]，可多次")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=50)
    args = ap.parse_args()

    shapes = [parse_shape(s) for s in args.shape] if args.shape else DEFAULT_SHAPES
    if args.cmd in ("dump", "all"):
        print(f"=== dump to {OUT_ROOT} ===")
        for (B, S, H, D, causal) in shapes:
            dump_case(B, S, H, D, causal, args.dtype)
    if args.cmd in ("bench", "all"):
        print("=== bench (CUPTI device time) ===")
        timer = CudaTimer(args.warmup, args.repeat)
        for (B, S, H, D, causal) in shapes:
            bench_case(B, S, H, D, causal, args.dtype, timer)


if __name__ == "__main__":
    main()
