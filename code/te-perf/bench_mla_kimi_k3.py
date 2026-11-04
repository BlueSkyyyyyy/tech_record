# Copyright (c) 2026. All rights reserved.
#
# Benchmark TransformerEngine fused attention on **Kimi-K3's MLA** core
# attention, forward + backward (training path).
#
# Kimi-K3 (config.json / modeling_kimi_linear.KimiMLAAttention):
#   num_attention_heads = 96, qk_nope = 128, qk_rope = 64 -> qk head dim = 192,
#   v_head_dim = 128, kv_lora_rank = 512, q_lora_rank = 1536, 24 MLA layers.
#   MLA uses **NoPE** in K3 (rotary_emb = None), so the 64 "rope" slots are
#   carried through unrotated.  The fused-attention workload is therefore a
#   plain causal softmax over per-head q/k of dim 192 and v of dim 128:
#       (batch, seqlen, num_heads, qk_head_dim, v_head_dim) = (b, s, 96, 192, 128)
#   which is exactly the 5-tuple MLA form already used for Kimi-K2.6 in
#   bench_te.py; only the head count (64 -> 96) changes.
#
# TE support (verified with tex.get_fused_attn_backend + a real run on H100 /
# TE 2.14): F16_arbitrary_seqlen is returned for training=True and
# training=False, bf16 and fp16, seqlen 256..8192.  v_head_dim=128 is *not*
# padded to 192 (FA2 in the HF model pads it to qk dim), so the PV matmul does
# 128/192 of the FLOPs a padded FA2 kernel would do.
#
# Timing: same methodology as bench_te.py -- pure device kernel time via CUPTI
# (torch.profiler), launch overhead excluded, warmup + repeat, mean over calls.
# The core timing/accounting helpers are imported from bench_te.py so both
# scripts report identical numbers.
#
# Usage (inside the kimi26_train container, on an idle GPU):
#   CUDA_VISIBLE_DEVICES=0 python bench_mla_kimi_k3.py --check
#   CUDA_VISIBLE_DEVICES=0 python bench_mla_kimi_k3.py --warmup 10 --repeat 50 \
#       --csv mla_kimi_k3.csv
#   CUDA_VISIBLE_DEVICES=0 python bench_mla_kimi_k3.py \
#       --seqs 1024,4096,16384 --dtypes bf16
#
# Notes / caveats:
#   * This is the *core attention* kernel only (QK^T -> softmax -> PV).  The
#     MLA projections (q_a/q_b, kv_a/kv_b, o_proj) and K3's full-rank output
#     gate are separate GEMMs and are not part of TE's fused_attn.
#   * K3 keeps the attention output in FP32 during training (biased-rounding
#     fix).  TE's fused_attn returns the output in the input dtype (bf16/fp16);
#     the kernel still accumulates softmax/PV in FP32 internally, but a
#     bit-exact FP32 output tensor is not exposed by this API.
#   * FLOPs follow the repo convention: fwd = 2*b*s*h*s*(qk+v), bwd = 2x fwd,
#     i.e. the full S^2 matmul.  A causal kernel skips ~half of the S^2 tiles,
#     so the printed %TC is an S^2-equivalent throughput, i.e. an *upper* bound
#     on real tensor-core utilization (real util ~= 0.5x printed), same
#     accounting convention as the other attention rows in this repo.

from __future__ import annotations

import argparse
import csv
import math
import sys

import torch

from bench_te import (
    H100,
    KernelTimer,
    attn_dims,
    bench_fused_attn,
    print_sep,
)

# Kimi-K3 MLA core-attention geometry (from ~/models/Kimi-K3/config.json).
K3_HEADS = 96
K3_QK_DIM = 192   # qk_nope_head_dim 128 + qk_rope_head_dim 64 (NoPE: unrotated)
K3_V_DIM = 128
K3_MLA_LAYERS = 24  # full_attn_layers in linear_attn_config


def k3_shape(batch: int, seqlen: int):
    return (batch, seqlen, K3_HEADS, K3_QK_DIM, K3_V_DIM)


def check(gpu: int = 0) -> None:
    """One-shot numerical check of TE MLA fwd + bwd against a torch reference.

    Reference math (fp32, same causal mask, scale = 1/sqrt(192)):
        S = Q K^T / sqrt(d) ; P = softmax(mask(S)) ; O = P V
    Gradients are obtained with torch autograd from that same graph.
    """
    import torch.nn.functional as F

    from transformer_engine.pytorch.constants import TE_DType
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        FusedAttnBackend,
        fused_attn_bwd,
        fused_attn_fwd,
    )

    torch.manual_seed(0)
    bs, s, nh, qk, vd = 1, 512, K3_HEADS, K3_QK_DIM, K3_V_DIM
    dt = torch.bfloat16
    q = torch.randn(bs * s, nh, qk, device="cuda", dtype=dt)
    k = torch.randn(bs * s, nh, qk, device="cuda", dtype=dt)
    v = torch.randn(bs * s, nh, vd, device="cuda", dtype=dt)
    cu = torch.arange(0, (bs + 1) * s, s, dtype=torch.int32, device="cuda")
    backend = FusedAttnBackend["F16_arbitrary_seqlen"]
    kw = dict(attn_bias_type="no_bias", attn_mask_type="causal",
              softmax_type="vanilla", qkv_layout="bshd_bshd_bshd")

    out, aux = fused_attn_fwd(True, s, s, cu, cu, q, k, v, dt, backend, None, **kw)
    assert tuple(out.shape) == (bs * s, nh, vd), out.shape

    # reference forward (fp32)
    qf = q.view(bs, s, nh, qk).permute(0, 2, 1, 3).float().requires_grad_(True)
    kf = k.view(bs, s, nh, qk).permute(0, 2, 1, 3).float().requires_grad_(True)
    vf = v.view(bs, s, nh, vd).permute(0, 2, 1, 3).float().requires_grad_(True)
    mask = torch.triu(torch.ones(s, s, device="cuda", dtype=torch.bool), 1)
    # scale -> mask -> softmax -> PV, all fp32
    logits = (qf @ kf.transpose(-1, -2)) / math.sqrt(qk)
    ref = torch.softmax(logits.masked_fill(mask, float("-inf")), -1) @ vf

    def cos(a, b):
        return F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1),
                                   dim=0).item()

    out_ref = out.view(bs, s, nh, vd).permute(0, 2, 1, 3)
    err = (out_ref.float() - ref).abs().max().item()
    c_fwd = cos(out_ref, ref)
    print(f"fwd: max abs err = {err:.3e}, cos = {c_fwd:.7f} "
          f"(|O| mean = {ref.abs().mean().item():.3e})")

    d_out = torch.randn_like(out)
    dq, dk, dv, *_ = fused_attn_bwd(
        s, s, cu, cu, q, k, v, out, d_out, dt,
        dqkv_dtype=TE_DType[dt], aux_ctx_tensors=list(aux),
        fused_attention_backend=backend, **kw,
    )
    ref.backward(d_out.view(bs, s, nh, vd).permute(0, 2, 1, 3).float())

    dq_t = dq.view(bs, s, nh, qk).permute(0, 2, 1, 3)
    dk_t = dk.view(bs, s, nh, qk).permute(0, 2, 1, 3)
    dv_t = dv.view(bs, s, nh, vd).permute(0, 2, 1, 3)
    c_dq, c_dk, c_dv = cos(dq_t, qf.grad), cos(dk_t, kf.grad), cos(dv_t, vf.grad)
    print(f"bwd: cos(dq) = {c_dq:.7f}, cos(dk) = {c_dk:.7f}, cos(dv) = {c_dv:.7f}")
    print(f"     max abs err: dq { (dq_t - qf.grad).abs().max().item():.3e}, "
          f"dk { (dk_t - kf.grad).abs().max().item():.3e}, "
          f"dv { (dv_t - vf.grad).abs().max().item():.3e}")

    assert c_fwd > 0.999, f"fwd mismatch: cos={c_fwd}"
    assert min(c_dq, c_dk, c_dv) > 0.999, "bwd mismatch"
    print("CHECK OK: TE fused_attn fwd+bwd matches the torch MLA reference.\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="TE fused_attn benchmark for Kimi-K3 MLA (fwd + bwd)")
    p.add_argument("--check", action="store_true",
                   help="run the numerical check against a torch reference, then exit")
    p.add_argument("--seqs", type=str,
                   default="1024,2048,4096,8192,16384",
                   help="comma-separated seqlen list (default: 1024,2048,4096,8192,16384)")
    p.add_argument("--extra-batches", type=str, default="2x4096",
                   help="extra batch x seq points, e.g. '2x4096,4x2048' (default: 2x4096)")
    p.add_argument("--dtypes", type=str, default="bf16,fp16",
                   help="comma-separated subset of bf16,fp16 (default: both)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--gpu", type=int, default=0,
                   help="informational only: which physical GPU you pinned via CUDA_VISIBLE_DEVICES")
    args = p.parse_args()

    if args.check:
        check(args.gpu)
        sys.exit(0)

    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"(CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")
    prop = torch.cuda.get_device_properties(0)
    print(f"SMs: {prop.multi_processor_count}, CC: {prop.major}.{prop.minor}, "
          f"mem {prop.total_memory / 2**30:.1f} GiB")
    print(f"Kimi-K3 MLA: heads={K3_HEADS}, qk={K3_QK_DIM} (nope128+rope64, NoPE), "
          f"v={K3_V_DIM}, layers={K3_MLA_LAYERS}; causal, no_bias, dropout=0")

    dt_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    dtypes = [dt_map[x.strip()] for x in args.dtypes.split(",") if x.strip()]

    shapes = []
    for s in [int(x) for x in args.seqs.split(",") if x.strip()]:
        shapes.append(k3_shape(1, s))
    if args.extra_batches:
        for item in args.extra_batches.split(","):
            item = item.strip()
            if not item:
                continue
            b, s = item.lower().split("x")
            shapes.append(k3_shape(int(b), int(s)))

    timer = KernelTimer(warmup=args.warmup, repeat=args.repeat)
    print(f"warmup={args.warmup}, repeat={args.repeat}; "
          f"time = CUPTI pure device kernel time (launch excluded)")

    results = []

    # ---- forward -----------------------------------------------------------
    print_sep("KIMI-K3 MLA — FUSED ATTENTION FORWARD (training=True)")
    print(f"{'shape (b,s,h,qk,v)':<26} {'dtype':<9} {'time(us)':>9} {'TFLOPS':>9} "
          f"{'%TC(S^2)':>8} {'GB/s':>10} {'AI':>7}")
    for shape in shapes:
        bs, s, nh, qk, vd, _ = attn_dims(shape)
        for dt in dtypes:
            r = bench_fused_attn(shape, dt, timer, do_backward=False)
            results.append(("mla_k3_fwd", shape, str(dt), r))
            flops = 2 * bs * s * nh * s * (qk + vd)
            ai = flops / (r["gb"] * 1e9)
            print(f"{str(shape):<26} {str(dt):<9} {r['kernel_ms']*1e3:>8.1f} "
                  f"{r['tflops']:>9.1f} {r['tflops']/H100['fp16_tflops']*100:>5.1f}% "
                  f"{r['gb_s']:>10.1f} {ai:>7.1f}")

    # ---- backward ----------------------------------------------------------
    print_sep("KIMI-K3 MLA — FUSED ATTENTION BACKWARD (training=True)")
    print(f"{'shape (b,s,h,qk,v)':<26} {'dtype':<9} {'time(us)':>9} {'TFLOPS':>9} "
          f"{'%TC(S^2)':>8} {'GB/s':>10} {'AI':>7}")
    for shape in shapes:
        bs, s, nh, qk, vd, _ = attn_dims(shape)
        for dt in dtypes:
            r = bench_fused_attn(shape, dt, timer, do_backward=True)
            results.append(("mla_k3_bwd", shape, str(dt), r))
            flops = 4 * bs * s * nh * s * (qk + vd)
            ai = flops / (r["gb"] * 1e9)
            print(f"{str(shape):<26} {str(dt):<9} {r['kernel_ms']*1e3:>8.1f} "
                  f"{r['tflops']:>9.1f} {r['tflops']/H100['fp16_tflops']*100:>5.1f}% "
                  f"{r['gb_s']:>10.1f} {ai:>7.1f}")

    # ---- derived: one K3 layer and the whole model's 24 MLA layers --------
    print_sep("DERIVED: KIMI-K3 MLA LAYER / MODEL COST (bf16, seqlen=4096, batch=1)")
    print("per-layer = core attention kernel only; x24 = MLA layers of K3's 93 layers.")
    print("(reuses the seqlen=4096 rows measured above; no extra timing -> no re-run drift)")
    print(f"{'phase':<10} {'per-layer(ms)':>14} {'x24(ms)':>10}")
    # look up the already-measured (1, 4096, 96, 192, 128) bf16 rows
    key_shape = k3_shape(1, 4096)
    lookup = {(k, tuple(sh), dt): r for k, sh, dt, r in results}
    kf = ("mla_k3_fwd", key_shape, str(torch.bfloat16))
    kb = ("mla_k3_bwd", key_shape, str(torch.bfloat16))
    if kf in lookup and kb in lookup:
        rf, rb = lookup[kf], lookup[kb]
        print(f"{'fwd':<10} {rf['kernel_ms']:>14.3f} {rf['kernel_ms']*K3_MLA_LAYERS:>10.2f}")
        print(f"{'bwd':<10} {rb['kernel_ms']:>14.3f} {rb['kernel_ms']*K3_MLA_LAYERS:>10.2f}")
        print(f"{'fwd+bwd':<10} {rf['kernel_ms']+rb['kernel_ms']:>14.3f} "
              f"{(rf['kernel_ms']+rb['kernel_ms'])*K3_MLA_LAYERS:>10.2f}")
    else:
        print("  skipped: needs bf16 and seqlen=4096 in --seqs / --dtypes")

    print_sep("H100 ROOFLINE REFERENCE")
    print(f"FP16/BF16 tensor-core peak : {H100['fp16_tflops']:>8.1f} TFLOPS (dense)")
    print(f"HBM3 bandwidth             : {H100['bandwidth']/1e12:>8.2f} TB/s")
    print(f"Roofline ridge (FP16/BF16) : AI = "
          f"{H100['fp16_tflops']*1e12/H100['bandwidth']:>6.1f} FLOP/byte")
    print("K3 MLA AI is ~100+ FLOP/byte for s>=1k, so it is compute-bound;")
    print("TFLOPS/%TC count the full S^2 (repo convention); a causal kernel skips ~half")
    print("of the S^2 tiles, so real tensor-core utilization is ~0.5x the printed %TC.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kernel", "shape", "dtype", "kernel_ms", "gb_s", "gb", "tflops"])
            for kernel, shape, dt, r in results:
                w.writerow([kernel, shape, dt, r.get("kernel_ms"),
                            r.get("gb_s"), r.get("gb"), r.get("tflops")])
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
