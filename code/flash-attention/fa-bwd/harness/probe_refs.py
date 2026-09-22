#!/usr/bin/env python3
"""探测本机可用的 FA / TE 反向参考实现，并做一次数值对拍。

用来在写自己的 kernel 之前确认「参考实现能跑、数值对得上、接口怎么调」。
在 kernel_lab 容器里运行：
    docker exec kernel_lab python /home/xieminglin/proj/tech_record/code/flash-attention/fa-bwd/harness/probe_refs.py
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cuda"


def ref_attn_fwd_bwd(q, k, v, do, causal=True):
    """纯 PyTorch（fp32）参考：返回 (o, dq, dk, dv)。q/k/v/do: [B,S,H,D] float32。"""
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    B, S, H, D = q.shape
    qq = q.transpose(1, 2)  # [B,H,S,D]
    kk = k.transpose(1, 2)
    vv = v.transpose(1, 2)
    scale = 1.0 / math.sqrt(D)
    s = (qq @ kk.transpose(-1, -2)) * scale
    if causal:
        mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = (p @ vv).transpose(1, 2)
    o.backward(do)
    return o.detach(), q.grad, k.grad, v.grad


def try_flash_attn(q, k, v, do, causal=True):
    try:
        import flash_attn
        from flash_attn import flash_attn_func
    except Exception as e:  # noqa
        return None, f"flash_attn import failed: {e}"
    try:
        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        o = flash_attn_func(q2, k2, v2, causal=causal)
        o.backward(do)
        return (o.detach(), q2.grad, k2.grad, v2.grad), f"flash_attn {flash_attn.__version__}"
    except Exception as e:  # noqa
        return None, f"flash_attn run failed: {e}"


def try_te(q, k, v, do, causal=True):
    try:
        from transformer_engine.pytorch.cpp_extensions.fused_attn import (
            fused_attn_fwd,
            fused_attn_bwd,
        )
        from transformer_engine.pytorch.cpp_extensions.fused_attn import (
            FusedAttnBackend,
            QKVLayout,
        )
        from transformer_engine.pytorch.constants import TE_DType
    except Exception as e:  # noqa
        try:
            from transformer_engine.pytorch.cpp_extensions.fused_attn import (
                fused_attn_fwd,
                fused_attn_bwd,
                FusedAttnBackend,
                QKVLayout,
            )
            from transformer_engine.pytorch.constants import TE_DType
        except Exception as e2:  # noqa
            return None, f"TE import failed: {e} / {e2}"
    try:
        B, S, H, D = q.shape
        cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=DEV)
        backend = FusedAttnBackend["F16_arbitrary_seqlen"]
        dtype = q.dtype
        qf = q.reshape(B * S, H, D).contiguous()
        kf = k.reshape(B * S, H, D).contiguous()
        vf = v.reshape(B * S, H, D).contiguous()
        mask = "causal" if causal else "no_mask"
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
            attn_bias_type="no_bias", attn_mask_type=mask,
            softmax_type="vanilla",
        )
        dq, dk, dv = dqkv[0], dqkv[1], dqkv[2]
        o = out.reshape(B, S, H, D)
        return (o.detach(), dq.reshape(B, S, H, D), dk.reshape(B, S, H, D), dv.reshape(B, S, H, D)), "TE"
    except Exception as e:  # noqa
        return None, f"TE run failed: {e}"


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    B, S, H, D = 1, 512, 16, 128
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"case: B={B} S={S} H={H} D={D} causal=True")
    for dt in (torch.float16, torch.bfloat16):
        q = torch.randn(B, S, H, D, device=DEV, dtype=dt)
        k = torch.randn(B, S, H, D, device=DEV, dtype=dt)
        v = torch.randn(B, S, H, D, device=DEV, dtype=dt)
        do = torch.randn(B, S, H, D, device=DEV, dtype=dt)
        o_ref, dq_ref, dk_ref, dv_ref = ref_attn_fwd_bwd(
            q.float(), k.float(), v.float(), do.float()
        )
        print(f"\n=== {dt} ===")
        fa, msg = try_flash_attn(q, k, v, do)
        if fa:
            o, dq, dk, dv = fa
            print(f"  [{msg}] o {maxdiff(o, o_ref):.3e}  dq {maxdiff(dq, dq_ref):.3e}  "
                  f"dk {maxdiff(dk, dk_ref):.3e}  dv {maxdiff(dv, dv_ref):.3e}")
        else:
            print(f"  flash_attn: {msg}")
        te, msg = try_te(q, k, v, do)
        if te:
            o, dq, dk, dv = te
            print(f"  [{msg}] o {maxdiff(o, o_ref):.3e}  dq {maxdiff(dq, dq_ref):.3e}  "
                  f"dk {maxdiff(dk, dk_ref):.3e}  dv {maxdiff(dv, dv_ref):.3e}")
        else:
            print(f"  TE: {msg}")


if __name__ == "__main__":
    main()
