#!/usr/bin/env python3
"""ncu 专用：跑 TE FP8 反向 4 次（便于 --launch-skip 3 取稳态）。

用法：python te_fp8_ncu.py "1 4096 16 128 causal"
"""
import sys

import torch


def main():
    import transformer_engine  # noqa: F401  （先加载 TE 才能 import transformer_engine_torch）
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        FusedAttnBackend, fused_attn_bwd, fused_attn_fwd,
    )
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
    spec = sys.argv[1].split()
    B, S, H, D = (int(x) for x in spec[:4])
    Hkv = H
    causal = True
    for t in spec[4:]:
        if t.startswith("kv="):
            Hkv = int(t[3:])
        if t == "full":
            causal = False
    DEV = "cuda"
    nominal = torch.bfloat16
    e4m3, e5m2 = tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2

    def Q(dt):
        return Float8Quantizer(scale=torch.ones(1, device=DEV), amax=torch.zeros(1, device=DEV),
                               fp8_dtype=dt, rowwise=True, columnwise=False)
    qkv_q, s_q, o_q = Q(e4m3), Q(e4m3), Q(e4m3)
    do_q, dp_q, dqkv_q = Q(e5m2), Q(e5m2), Q(e5m2)
    torch.manual_seed(0)
    cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=DEV)
    q = torch.randn(B * S, H, D, device=DEV, dtype=nominal)
    k = torch.randn(B * S, Hkv, D, device=DEV, dtype=nominal)
    v = torch.randn(B * S, Hkv, D, device=DEV, dtype=nominal)
    do = torch.randn(B * S, H, D, device=DEV, dtype=nominal)
    q8, k8, v8, do8 = qkv_q(q), qkv_q(k), qkv_q(v), do_q(do)
    mask = "causal" if causal else "no_mask"
    backend = FusedAttnBackend["FP8"]
    for _ in range(4):
        out, aux, *_ = fused_attn_fwd(
            True, S, S, cu, cu, q8, k8, v8, nominal, backend, None,
            s_quantizer=s_q, o_quantizer=o_q, attn_bias_type="no_bias",
            attn_mask_type=mask, softmax_type="vanilla", qkv_layout="bshd_bshd_bshd")
        fused_attn_bwd(S, S, cu, cu, q8, k8, v8, out, do8, nominal, do8._fp8_dtype, list(aux),
                       backend, qkv_layout="bshd_bshd_bshd", s_quantizer=s_q, dp_quantizer=dp_q,
                       dqkv_quantizer=dqkv_q, attn_bias_type="no_bias",
                       attn_mask_type=mask, softmax_type="vanilla")
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
