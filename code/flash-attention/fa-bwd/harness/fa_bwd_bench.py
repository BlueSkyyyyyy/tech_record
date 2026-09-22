#!/usr/bin/env python3
"""FA 反向：参考值 dump + 数值对拍 + 性能基准（FA / TE 对照）。

用法（在 kernel_lab 容器内）：
  python fa_bwd_bench.py dump  --shape "1 1024 40 128 kv=8 causal" ...
  python fa_bwd_bench.py bench --dtype fp16 --shape "1 256 2 512 causal" ...
  python fa_bwd_bench.py all

shape 语法（空白或逗号分隔，逗号/括号会被忽略）：
  B S H D [kv=<num_kv_heads>] [Dv=<v_head_dim>] [causal|full]
  默认 kv=H（MHA）、Dv=D（普通 attention）；GQA/MQA 用 kv=，MLA 用 Dv=。

设计要点：
- 输入用固定 seed 生成，张量以 fp32 CPU npy 保存（fp16/bf16 可被 fp32 无损表示），
  方便自己的 kernel 直接 load 同一份输入、和同一份 ref 输出逐元素比对。
- ref = 纯 PyTorch fp32（autograd，支持 GQA/MQA + MLA 的 Dv≠D）；
  fa = flash_attn 2.7.4；te = TransformerEngine 2.14。
- 输出目录：/home/xieminglin/proj/output/fa-bwd/<case>/ ；meta.json 记录 shape/dtype/causal/seed。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

OUT_ROOT = Path("/home/xieminglin/proj/output/fa-bwd")
DEV = "cuda"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.bfloat16}

# 基础 MHA 形状（B, S, H, D, causal）
DEFAULT_SHAPES = [
    (1, 512, 16, 128, True),
    (1, 1024, 16, 128, True),
    (4, 2048, 16, 128, True),
    (2, 2048, 16, 128, False),
    (1, 4096, 16, 128, True),
]

# 用户指定：GQA / MQA / MLA（(B,S,H,D,causal,Hkv,Dv)）
REQUESTED_SHAPES = [
    (1, 1024, 40, 128, True, 8, 128),   # Qwen3-8B GQA kv=8
    (1, 1024, 32, 128, True, 4, 128),   # Qwen3-30B-A3B GQA kv=4
    (1, 1024, 64, 128, True, 4, 128),   # Qwen3-235B-A22B GQA kv=4
    (1, 1024, 64, 128, True, 1, 128),   # DeepSeek DSA indexer MQA kv=1
    (1, 256, 2, 512, True, 2, 512),     # MLA head_dim=512
    (1, 512, 4, 512, True, 4, 512),     # MLA head_dim=512
    (1, 1024, 2, 512, True, 2, 512),    # MLA head_dim=512
]


def norm_spec(B, S, H, D, causal, Hkv=None, Dv=None):
    Hkv = H if Hkv is None else Hkv
    Dv = D if Dv is None else Dv
    return dict(B=B, S=S, H=H, D=D, causal=causal, Hkv=Hkv, Dv=Dv)


def case_slug(sh, dtype):
    s = f"b{sh['B']}_s{sh['S']}_h{sh['H']}_d{sh['D']}"
    if sh["Hkv"] != sh["H"]:
        s += f"_kv{sh['Hkv']}"
    if sh["Dv"] != sh["D"]:
        s += f"_dv{sh['Dv']}"
    return s + f"_{'causal' if sh['causal'] else 'full'}_{dtype}"


def seed_case(sh):
    return 1234 + sh["B"] * 31 + sh["S"] * 7 + sh["H"] * 3 + sh["D"] + sh.get("Hkv", 0) * 5


# ----------------------------- 参考实现 -----------------------------
def ref_attn(q, k, v, do, causal=True):
    """fp32 autograd 参考，支持 GQA/MQA 与 Dv≠D（MLA）。
    q:[B,S,H,D]  k:[B,S,Hkv,D]  v:[B,S,Hkv,Dv]  do:[B,S,H,Dv]
    返回 o:[B,S,H,Dv], dq, dk, dv。"""
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    B, S, H, D = q.shape
    Hkv = k.shape[2]
    Dv = v.shape[-1]
    qq = q.transpose(1, 2)                                  # [B,H,S,D]
    kk = k.transpose(1, 2)                                  # [B,Hkv,S,D]
    vv = v.transpose(1, 2)                                  # [B,Hkv,S,Dv]
    if Hkv != H:                                            # GQA/MQA 广播 KV 头
        rep = H // Hkv
        kk = kk.repeat_interleave(rep, dim=1)
        vv = vv.repeat_interleave(rep, dim=1)
    scale = 1.0 / math.sqrt(D)
    s = (qq @ kk.transpose(-1, -2)) * scale
    if causal:
        mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = (p @ vv).transpose(1, 2)                            # [B,S,H,Dv]
    o.backward(do)
    return o.detach(), q.grad, k.grad, v.grad


def fa_bwd(q, k, v, do, causal=True):
    """flash_attn 2.7.4 反向。GQA/MQA 支持；Dv≠D（MLA）FA2 一般不支持。"""
    from flash_attn import flash_attn_func
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    o = flash_attn_func(q2, k2, v2, causal=causal)
    o.backward(do)
    return o.detach(), q2.grad, k2.grad, v2.grad


def te_bwd(q, k, v, do, causal=True):
    """TE 2.14 反向。支持 GQA/MQA；MLA（Dv≠D）需 qk≤..，head_dim=512 训练 bwd 不支持。"""
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        FusedAttnBackend, fused_attn_bwd, fused_attn_fwd,
    )
    from transformer_engine.pytorch.constants import TE_DType
    B, S, H, D = q.shape
    Hkv, Dv = k.shape[2], v.shape[-1]
    cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=DEV)
    backend = FusedAttnBackend["F16_arbitrary_seqlen"]
    dtype = q.dtype
    mask = "causal" if causal else "no_mask"
    qf = q.reshape(B * S, H, D).contiguous()
    kf = k.reshape(B * S, Hkv, D).contiguous()
    vf = v.reshape(B * S, Hkv, Dv).contiguous()
    out, aux = fused_attn_fwd(
        True, S, S, cu, cu, qf, kf, vf, dtype, backend, None,
        attn_bias_type="no_bias", attn_mask_type=mask,
        softmax_type="vanilla", qkv_layout="bshd_bshd_bshd",
    )
    dof = do.reshape(B * S, H, Dv).contiguous()
    dqkv = fused_attn_bwd(
        S, S, cu, cu, qf, kf, vf, out, dof, dtype,
        qkv_layout="bshd_bshd_bshd", dqkv_dtype=TE_DType[dtype],
        aux_ctx_tensors=list(aux), fused_attention_backend=backend,
        attn_bias_type="no_bias", attn_mask_type=mask, softmax_type="vanilla",
    )
    return (out.reshape(B, S, H, Dv), dqkv[0].reshape(B, S, H, D),
            dqkv[1].reshape(B, S, Hkv, D), dqkv[2].reshape(B, S, Hkv, Dv))


def _fp8_q(fp8_dtype, tex):
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
    return Float8Quantizer(scale=torch.ones(1, device=DEV), amax=torch.zeros(1, device=DEV),
                           fp8_dtype=fp8_dtype, rowwise=True, columnwise=False)


def te_bwd_fp8(q, k, v, do, causal=True):
    """TE FP8 反向（Q/K/V/S/O=E4M3，dO/dP/dQKV=E5M2 rowwise）；返回反量化后的 fp32 结果。"""
    import transformer_engine  # noqa: F401  先导主包，transformer_engine_torch 才可被找到
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        FusedAttnBackend, fused_attn_bwd, fused_attn_fwd,
    )
    B, S, H, D = q.shape
    Hkv, Dv = k.shape[2], v.shape[-1]
    nominal = torch.bfloat16
    e4m3, e5m2 = tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2
    cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=DEV)
    qkv_q, s_q, o_q = _fp8_q(e4m3, tex), _fp8_q(e4m3, tex), _fp8_q(e4m3, tex)
    do_q, dp_q, dqkv_q = _fp8_q(e5m2, tex), _fp8_q(e5m2, tex), _fp8_q(e5m2, tex)
    qf = q.reshape(B * S, H, D).to(nominal).contiguous()
    kf = k.reshape(B * S, Hkv, D).to(nominal).contiguous()
    vf = v.reshape(B * S, Hkv, Dv).to(nominal).contiguous()
    dof = do.reshape(B * S, H, Dv).to(nominal).contiguous()
    q8, k8, v8, do8 = qkv_q(qf), qkv_q(kf), qkv_q(vf), do_q(dof)
    backend = FusedAttnBackend["FP8"]
    mask = "causal" if causal else "no_mask"
    out, aux, *_ = fused_attn_fwd(
        True, S, S, cu, cu, q8, k8, v8, nominal, backend, None,
        s_quantizer=s_q, o_quantizer=o_q, attn_bias_type="no_bias",
        attn_mask_type=mask, softmax_type="vanilla", qkv_layout="bshd_bshd_bshd")
    dqkv = fused_attn_bwd(
        S, S, cu, cu, q8, k8, v8, out, do8, nominal, do8._fp8_dtype, list(aux), backend,
        qkv_layout="bshd_bshd_bshd", s_quantizer=s_q, dp_quantizer=dp_q, dqkv_quantizer=dqkv_q,
        attn_bias_type="no_bias", attn_mask_type=mask, softmax_type="vanilla")

    def dq_(t):
        return (t.dequantize() if hasattr(t, "dequantize") else t).float()
    return (dq_(out).reshape(B, S, H, Dv), dq_(dqkv[0]).reshape(B, S, H, D),
            dq_(dqkv[1]).reshape(B, S, Hkv, D), dq_(dqkv[2]).reshape(B, S, Hkv, Dv))


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def relmax(a, b):
    d = (a.float() - b.float()).abs()
    scale = b.float().abs().clamp_min(1e-3)
    return (d / scale).max().item()


# ----------------------------- dump -----------------------------
def dump_case(sh, dtype_name):
    dtype = DTYPES[dtype_name]
    torch.manual_seed(seed_case(sh))
    B, S, H, D, Hkv, Dv = sh["B"], sh["S"], sh["H"], sh["D"], sh["Hkv"], sh["Dv"]
    q = torch.randn(B, S, H, D, device=DEV, dtype=dtype)
    k = torch.randn(B, S, Hkv, D, device=DEV, dtype=dtype)
    v = torch.randn(B, S, Hkv, Dv, device=DEV, dtype=dtype)
    do = torch.randn(B, S, H, Dv, device=DEV, dtype=dtype)

    o_ref, dq_ref, dk_ref, dv_ref = ref_attn(
        q.float(), k.float(), v.float(), do.float(), sh["causal"])
    res = {"ref": (o_ref, dq_ref, dk_ref, dv_ref)}
    if dtype_name == "fp8":          # FP8 只有 TE（FA 无反向 FP8）
        backends = [("te", te_bwd_fp8)]
    else:
        backends = [("fa", fa_bwd), ("te", te_bwd)]
    for who, fn in backends:
        try:
            res[who] = fn(q, k, v, do, sh["causal"])
        except Exception as e:  # noqa
            print(f"  [{who}] failed: {str(e)[:160]}")

    slug = case_slug(sh, dtype_name)
    d = OUT_ROOT / slug
    d.mkdir(parents=True, exist_ok=True)

    def save(name, t):
        np.save(d / f"{name}.npy", t.detach().float().cpu().numpy())

    for name, t in (("q", q), ("k", k), ("v", v), ("do", do)):
        save(name, t)
    for who, (o, dq, dk, dv) in res.items():
        for name, t in (("o", o), ("dq", dq), ("dk", dk), ("dv", dv)):
            save(f"{who}_{name}", t)
    (d / "meta.json").write_text(json.dumps({
        **sh, "dtype": dtype_name, "seed": seed_case(sh), "scale": 1.0 / math.sqrt(D),
        "which": list(res.keys()), "slug": slug,
    }, indent=2, ensure_ascii=False))

    lines = [f"[{slug}]"]
    for who in ("fa", "te"):
        if who in res:
            lines.append(f"  {who:3s} vs ref: o {maxdiff(res[who][0], o_ref):.2e} "
                         f"dq {maxdiff(res[who][1], dq_ref):.2e} "
                         f"dk {maxdiff(res[who][2], dk_ref):.2e} "
                         f"dv {maxdiff(res[who][3], dv_ref):.2e}")
    print("\n".join(lines))
    return slug


# ----------------------------- bench -----------------------------
class CudaTimer:
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
        return sum(e.device_time_total for e in prof.key_averages()) / self.repeat / 1e3


def bench_case(sh, dtype_name, timer):
    dtype = DTYPES[dtype_name]
    torch.manual_seed(seed_case(sh))
    B, S, H, D, Hkv, Dv = sh["B"], sh["S"], sh["H"], sh["D"], sh["Hkv"], sh["Dv"]
    q = torch.randn(B, S, H, D, device=DEV, dtype=dtype)
    k = torch.randn(B, S, Hkv, D, device=DEV, dtype=dtype)
    v = torch.randn(B, S, Hkv, Dv, device=DEV, dtype=dtype)
    do = torch.randn(B, S, H, Dv, device=DEV, dtype=dtype)
    flops = 4.0 * B * S * H * S * (D + Dv)  # bwd ≈ 2x fwd
    tag = case_slug(sh, dtype_name)
    cells = []
    pairs = [("te", te_bwd_fp8)] if dtype_name == "fp8" else [("fa", fa_bwd), ("te", te_bwd)]
    for who, fn in pairs:
        try:
            ms = timer.device_time(lambda: fn(q, k, v, do, sh["causal"]))
            cells.append(f"{who}={ms:.4f}ms/{flops / (ms * 1e-3) / 1e12:.2f}TF")
        except Exception as e:  # noqa
            cells.append(f"{who}=NA({str(e)[:40]})")
    print(f"[{tag}] " + "  ".join(cells))
    return tag, cells


def parse_shape(s):
    """解析 "1 1024 40 128 kv=8 causal" / "(1,1024,40,128) kv=8 full" 等。"""
    toks = s.replace(",", " ").replace("(", " ").replace(")", " ").split()
    nums, causal, Hkv, Dv = [], True, None, None
    for t in toks:
        tl = t.lower()
        if tl in ("causal", "full"):
            causal = tl == "causal"
        elif "=" in t:
            key, val = t.split("=", 1)
            key = key.lower()
            if key in ("kv", "hkv", "kv_heads", "num_kv_heads"):
                Hkv = int(val)
            elif key in ("dv", "v_head_dim", "vdim"):
                Dv = int(val)
            elif key in ("b", "s", "h", "d"):
                nums.append(int(val))
            else:
                raise ValueError(f"unknown key {t}")
        else:
            nums.append(int(t))
    if len(nums) < 4:
        raise ValueError(f"shape needs B S H D: {s}")
    B, S, H, D = nums[:4]
    return norm_spec(B, S, H, D, causal, Hkv, Dv)


def parse_tuple(t):
    """(B,S,H,D,causal,Hkv,Dv) -> dict（供内置列表用）。"""
    B, S, H, D, causal = t[:5]
    Hkv = t[5] if len(t) > 5 else H
    Dv = t[6] if len(t) > 6 else D
    return norm_spec(B, S, H, D, causal, Hkv, Dv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "bench", "all"])
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--shape", nargs="+", action="append", default=None)
    ap.add_argument("--requested", action="store_true", help="跑用户指定的 GQA/MQA/MLA 形状集")
    ap.add_argument("--dtypes", nargs="+", default=None, help="对多个 dtype 依次跑")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=50)
    args = ap.parse_args()

    if args.shape:
        shapes = [parse_shape(" ".join(s)) for s in args.shape]
    elif args.requested:
        shapes = [parse_tuple(t) for t in REQUESTED_SHAPES]
    else:
        shapes = [parse_tuple(t) for t in DEFAULT_SHAPES]

    dtypes = args.dtypes or [args.dtype]
    for dt in dtypes:
        if args.cmd in ("dump", "all"):
            print(f"=== dump to {OUT_ROOT} (dtype={dt}) ===")
            for sh in shapes:
                dump_case(sh, dt)
        if args.cmd in ("bench", "all"):
            print(f"=== bench CUPTI device time (dtype={dt}) ===")
            timer = CudaTimer(args.warmup, args.repeat)
            for sh in shapes:
                bench_case(sh, dt, timer)


if __name__ == "__main__":
    main()
