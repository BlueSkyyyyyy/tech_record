#!/usr/bin/env python3
"""ncu 专用驱动：只跑 FA 或 TE 的反向（跑 4 次，便于 ncu --launch-skip 3 取稳态）。

用法：python fa_vs_te_ncu.py {fa|te} "B S H D [kv=..] [causal|full]" [fp16|bf16]
"""
import sys

import torch

DEV = "cuda"
DT = {"fp16": torch.float16, "bf16": torch.bfloat16}


def main():
    who = sys.argv[1]
    spec = sys.argv[2].split()
    dt = DT[sys.argv[3] if len(sys.argv) > 3 else "fp16"]
    B, S, H, D = (int(x) for x in spec[:4])
    Hkv, causal = H, True
    for t in spec[4:]:
        if t.startswith("kv="):
            Hkv = int(t[3:])
        if t == "full":
            causal = False
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device=DEV, dtype=dt)
    k = torch.randn(B, S, Hkv, D, device=DEV, dtype=dt)
    v = torch.randn(B, S, Hkv, D, device=DEV, dtype=dt)
    do = torch.randn(B, S, H, D, device=DEV, dtype=dt)

    if who == "fa3":
        from flash_attn_3 import flash_attn_interface as f3
        for _ in range(4):
            q2 = q.clone().requires_grad_(True)
            k2 = k.clone().requires_grad_(True)
            v2 = v.clone().requires_grad_(True)
            o = f3.flash_attn_func(q2, k2, v2, causal=causal)
            o.backward(do)
    elif who == "fa":
        from flash_attn import flash_attn_func
        for _ in range(4):
            q2 = q.clone().requires_grad_(True)
            k2 = k.clone().requires_grad_(True)
            v2 = v.clone().requires_grad_(True)
            o = flash_attn_func(q2, k2, v2, causal=causal)
            o.backward(do)
    else:
        from transformer_engine.pytorch.cpp_extensions.fused_attn import (
            FusedAttnBackend, fused_attn_bwd, fused_attn_fwd,
        )
        from transformer_engine.pytorch.constants import TE_DType
        cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=DEV)
        qf = q.reshape(B * S, H, D).contiguous()
        kf = k.reshape(B * S, Hkv, D).contiguous()
        vf = v.reshape(B * S, Hkv, D).contiguous()
        dof = do.reshape(B * S, H, D).contiguous()
        mask = "causal" if causal else "no_mask"
        backend = FusedAttnBackend["F16_arbitrary_seqlen"]
        for _ in range(4):
            out, aux = fused_attn_fwd(True, S, S, cu, cu, qf, kf, vf, dt, backend, None,
                                      attn_bias_type="no_bias", attn_mask_type=mask,
                                      softmax_type="vanilla", qkv_layout="bshd_bshd_bshd")
            fused_attn_bwd(S, S, cu, cu, qf, kf, vf, out, dof, dt,
                           qkv_layout="bshd_bshd_bshd", dqkv_dtype=TE_DType[dt],
                           aux_ctx_tensors=list(aux), fused_attention_backend=backend,
                           attn_bias_type="no_bias", attn_mask_type=mask, softmax_type="vanilla")
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
