# Copyright (c) 2026. All rights reserved.
#
# Benchmark TransformerEngine kernels:
#   - te.rmsnorm (rmsnorm_fwd)
#   - te.rmsnorm_bwd
#   - te.fused_attn_fwd / te.fused_attn_bwd
#
# Measures achieved throughput (TFLOPS / GB/s) via CUDA events and compares to
# hard-coded roofline numbers for NVIDIA H100 SXM (132 SM, 80GB HBM3).
#
# Usage (run inside the kimi26_train container, on a machine owning the GPU):
#   python bench_te.py                # run everything, no correctness check
#   python bench_te.py --check        # run a one-shot correctness check first
#   python bench_te.py --rmsnorm-only # only rmsnorm / rmsnorm_bwd
#   python bench_te.py --attn-only    # only fused_attn fwd/bwd
#   python bench_te.py --csv out.csv  # write results to CSV
#
# Environment flags that influence kernel backend (kept at TE defaults):
#   NVTE_FUSED_ATTN=1      (use fused attention)
#   NVTE_FUSED_ATTN_USE_FAv2_BWD=0  (use FAv2 backward on cc9.0)
#   NVTE_BIAS_DROPOUT_FUSION=1  (fuse bias+dropout)

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch

# --------------------------------------------------------------------------- #
# H100 SXM roofline constants (hard-coded, device-agnostic reference).
# --------------------------------------------------------------------------- #
#   132 SMs @ 1.980 GHz
#   FP16/BF16 tensor-core dense: ~989.4 TFLOPS (no sparsity)
#   FP16/BF16 with 2:4 sparsity: ~1978.8 TFLOPS
#   FP32 (CUDA core FMA):       ~66.9  TFLOPS
#   HBM3 bandwidth:             ~3.35  TB/s (3.35e12 B/s)
H100 = {
    "fp16_tflops": 989.4,
    "bf16_tflops": 989.4,
    "fp32_tflops": 66.9,
    "bandwidth": 3.35e12,  # bytes per second
}

# --------------------------------------------------------------------------- #
# TE imports (must run inside container with transformer_engine installed)
# --------------------------------------------------------------------------- #
try:
    import transformer_engine.pytorch as te  # noqa: F401  (ensures path setup)
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.constants import TE_DType
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        fused_attn_fwd,
        fused_attn_bwd,
        FusedAttnBackend,
    )
except Exception as e:  # pragma: no cover - import at bench time
    print(f"[FATAL] unable to import transformer_engine: {e}", file=sys.stderr)
    sys.exit(1)

from transformer_engine_torch import rmsnorm_fwd, rmsnorm_bwd, rmsnorm_bwd_add  # noqa: E402


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
class KernelTimer:
    """CUDA-event timer with warmup + repeat, returns mean/median latency (ms)."""

    def __init__(self, warmup: int = 10, repeat: int = 50):
        self.warmup = warmup
        self.repeat = repeat
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def time(self, fn, *args, **kwargs):
        for _ in range(self.warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        times = []
        for _ in range(self.repeat):
            self.start.record()
            fn(*args, **kwargs)
            self.end.record()
            torch.cuda.synchronize()
            times.append(self.start.elapsed_time(self.end))
        times = sorted(times)
        mean = sum(times) / len(times)
        med = times[len(times) // 2]
        return mean, med, times


def bytes_of(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


# --------------------------------------------------------------------------- #
# rmsnorm fwd
# --------------------------------------------------------------------------- #
def bench_rmsnorm_fwd(shape, dtype, timer):
    rows, cols = shape
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    w = torch.randn(cols, device="cuda", dtype=dtype)
    # rmsnorm_fwd(input, weight, eps, out(None), quantizer(None), out_dtype,
    #             sm_margin, zero_centered_gamma)
    def fn():
        y, _, _ = rmsnorm_fwd(x, w, 1e-5, None, None, TE_DType[dtype], 0, False)
        return y

    mean, med, _ = timer.time(fn)
    ms = med
    # bytes moved: read x + read w + write y (+ rsigma, negligible)
    gb = (bytes_of(x) + bytes_of(w) * 1 + bytes_of(x)) / 1e9
    gb_s = gb / (ms / 1e3)
    return {"mean_ms": mean, "med_ms": med, "gb_s": gb_s, "gb": gb}


# --------------------------------------------------------------------------- #
# rmsnorm bwd: needs fwd to produce y (which carries rsigma) then call bwd
# --------------------------------------------------------------------------- #
def bench_rmsnorm_bwd(shape, dtype, timer):
    rows, cols = shape
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    w = torch.randn(cols, device="cuda", dtype=dtype)
    dy = torch.randn(rows, cols, device="cuda", dtype=dtype)

    # forward to get rsigma
    y, _, rsigma = rmsnorm_fwd(x, w, 1e-5, None, None, TE_DType[dtype], 0, False)

    # rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma)
    def fn():
        dx, dw = rmsnorm_bwd(dy, x, rsigma, w, 0, False)
        return dx, dw

    mean, med, _ = timer.time(fn)
    ms = med
    # read dy + read x + read rsigma + read w, write dx + write dw
    gb = (bytes_of(dy) + bytes_of(x) + rows * 4 + bytes_of(w) + bytes_of(x) + bytes_of(w)) / 1e9
    gb_s = gb / (ms / 1e3)
    return {"mean_ms": mean, "med_ms": med, "gb_s": gb_s, "gb": gb}


# --------------------------------------------------------------------------- #
# rmsnorm bwd + add (fused backward of rmsnorm with a residual add):
#   forward:  z = rmsnorm(x) + add   (add = residual / extra output)
#   backward: fused kernel computes dx (incl. grad of add) + dw in one pass.
#   TE exposes this via rmsnorm_bwd_add(dz, x, add, rsigma, gamma, ...).
# --------------------------------------------------------------------------- #
def bench_rmsnorm_bwd_add(shape, dtype, timer):
    rows, cols = shape
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    w = torch.randn(cols, device="cuda", dtype=dtype)
    add = torch.randn(rows, cols, device="cuda", dtype=dtype)
    dy = torch.randn(rows, cols, device="cuda", dtype=dtype)

    # forward to get rsigma
    _, _, rsigma = rmsnorm_fwd(x, w, 1e-5, None, None, TE_DType[dtype], 0, False)

    # rmsnorm_bwd_add(dz, x, add, rsigma, gamma, sm_margin, zero_centered_gamma)
    def fn():
        dx, dw = rmsnorm_bwd_add(dy, x, add, rsigma, w, 0, False)
        return dx, dw

    mean, med, _ = timer.time(fn)
    ms = med
    # read dy + read x + read add + read rsigma + read w, write dx + write dw
    gb = (bytes_of(dy) + bytes_of(x) + bytes_of(add) + rows * 4
          + bytes_of(w) + bytes_of(x) + bytes_of(w)) / 1e9
    gb_s = gb / (ms / 1e3)
    return {"mean_ms": mean, "med_ms": med, "gb_s": gb_s, "gb": gb}


# --------------------------------------------------------------------------- #
# fused attention fwd / bwd  (training, causal, dropped 0)
# --------------------------------------------------------------------------- #
def bench_fused_attn(shape, dtype, timer, do_backward=False):
    bs, seqlen, num_heads, head_dim = shape

    q = torch.randn(bs * seqlen, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(bs * seqlen, num_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(bs * seqlen, num_heads, head_dim, device="cuda", dtype=dtype)

    cu_seqlens = torch.arange(0, (bs + 1) * seqlen, seqlen, dtype=torch.int32, device="cuda")
    max_seqlen = seqlen

    backend = FusedAttnBackend["F16_arbitrary_seqlen"]
    attn_bias_type = "no_bias"
    attn_mask_type = "causal"
    softmax_type = "vanilla"

    # forward
    out, aux_ctx = fused_attn_fwd(
        True, max_seqlen, max_seqlen, cu_seqlens, cu_seqlens,
        q, k, v, dtype, backend, None,
        attn_bias_type=attn_bias_type, attn_mask_type=attn_mask_type,
        softmax_type=softmax_type,
        qkv_layout="bshd_bshd_bshd",
    )

    d_out = torch.randn_like(out)

    if not do_backward:
        def fn_fwd():
            o, _ = fused_attn_fwd(
                True, max_seqlen, max_seqlen, cu_seqlens, cu_seqlens,
                q, k, v, dtype, backend, None,
                attn_bias_type=attn_bias_type, attn_mask_type=attn_mask_type,
                softmax_type=softmax_type, qkv_layout="bshd_bshd_bshd",
            )
            return o

        mean, med, _ = timer.time(fn_fwd)
        ms = med
        # q + k + v read, o write (approx; s/m intermediate ignored)
        gb = (bytes_of(q) * 3 + bytes_of(out)) / 1e9
        gb_s = gb / (ms / 1e3)
        # FLOPs: QK^T (2*b*s*h*s*d) + PV (2*b*s*h*s*d) = 4*b*s*h*s*d
        flops = 4 * bs * seqlen * num_heads * seqlen * head_dim
        tflops = flops / (ms / 1e3) / 1e12
        return {"mean_ms": mean, "med_ms": med, "gb_s": gb_s, "gb": gb, "tflops": tflops}

    # backward
    def fn_bwd():
        dqkv = fused_attn_bwd(
            max_seqlen, max_seqlen, cu_seqlens, cu_seqlens,
            q, k, v, out, d_out, dtype,
            qkv_layout="bshd_bshd_bshd",
            dqkv_dtype=TE_DType[dtype],
            aux_ctx_tensors=list(aux_ctx),
            fused_attention_backend=backend,
            attn_bias_type="no_bias", attn_mask_type="causal",
            softmax_type="vanilla",
        )
        return dqkv

    mean, med, _ = timer.time(fn_bwd)
    ms = med
    gb = (bytes_of(q) * 3 + bytes_of(out) + bytes_of(d_out) + bytes_of(q) * 3) / 1e9
    gb_s = gb / (ms / 1e3)
    # FLOPs: backward ~ 2x forward = 8*b*s*h*s*d
    flops = 8 * bs * seqlen * num_heads * seqlen * head_dim
    tflops = flops / (ms / 1e3) / 1e12
    return {"mean_ms": mean, "med_ms": med, "gb_s": gb_s, "gb": gb, "tflops": tflops}


# --------------------------------------------------------------------------- #
# Roofline helpers
# --------------------------------------------------------------------------- #
def roofline(arithmetic_intensity, peak_flops_s, mem_bw_b_s):
    """Attainable performance = min(peak_flops, AI * BW)."""
    return min(peak_flops_s, arithmetic_intensity * mem_bw_b_s)


def print_sep(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    p = argparse.ArgumentParser(description="TE rmsnorm / fused_attn benchmark")
    p.add_argument("--check", action="store_true", help="run one-shot correctness check first")
    p.add_argument("--rmsnorm-only", action="store_true")
    p.add_argument("--attn-only", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--csv", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(0)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    prop = torch.cuda.get_device_properties(0)
    print(f"SMs: {prop.multi_processor_count}, CC: {prop.major}.{prop.minor}")

    timer = KernelTimer(warmup=args.warmup, repeat=args.repeat)

    # rmsnorm shapes: (rows, cols) = (batch*seq, hidden)
    rms_shapes = [
        (128, 512),
        (256, 512),
        (64, 1024),
        (128, 1024),
        (512, 512),
        (1024, 512),
        (1024, 1024),
        (1024, 2048),
        (2048, 1024),
        (2048, 2048),
        (4096, 2048),
        (4096, 4096),
        (8192, 2048),
        (8192, 4096),
        (16384, 2048),
        (32768, 2048),
    ]
    dtypes = [torch.float32, torch.bfloat16, torch.float16]

    # attention shapes: (batch, seqlen, num_heads, head_dim)
    attn_shapes = [
        (1, 512, 16, 128),
        (1, 1024, 16, 128),
        (1, 2048, 16, 128),
        (2, 2048, 16, 128),
        (4, 2048, 16, 128),
        (8, 2048, 16, 128),
        (1, 4096, 16, 128),
        (2, 4096, 16, 128),
        (4, 1024, 32, 128),
        (8, 1024, 32, 128),
        (4, 8192, 16, 128),
    ]
    attn_dtypes = [torch.bfloat16, torch.float16]

    results = []

    # measure kernel launch overhead with a trivial (minimal) kernel
    empty = torch.zeros(1, device="cuda")
    def noop():
        empty.fill_(1.0)
    _, launch_ms, _ = timer.time(noop)
    print_sep("KERNEL LAUNCH OVERHEAD (trivial fill_)")
    print(f"minimal-kernel round-trip latency = {launch_ms*1e3:.1f} us/call")
    print("(this is the fixed cost mixed into every measured kernel time;")
    print(" subtract it to isolate the true kernel execution time for small shapes)")

    launch_ovh = launch_ms

    if args.check:
        print_sep("CORRECTNESS CHECK (one-shot)")
        # rmsnorm vs torch reference
        rows, cols = 128, 512
        x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(cols, device="cuda", dtype=torch.bfloat16)
        y, _, rsigma = rmsnorm_fwd(x, w, 1e-5, None, None, TE_DType[torch.bfloat16], 0, False)
        ref = x.float() / torch.sqrt((x.float() ** 2).mean(-1, keepdim=True) + 1e-5) * w.float()
        err = (y.float() - ref).abs().max().item()
        print(f"rmsnorm_fwd max abs err = {err:.3e} (out abs mean = {ref.abs().mean():.3e})")
        assert err < 0.1, "rmsnorm fwd mismatch"
        # attention vs torch MHA
        bs, s, nh, hd = 1, 256, 4, 64
        q = torch.randn(bs * s, nh, hd, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(bs * s, nh, hd, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(bs * s, nh, hd, device="cuda", dtype=torch.bfloat16)
        cu = torch.arange(0, (bs + 1) * s, s, dtype=torch.int32, device="cuda")
        out, _ = fused_attn_fwd(
            True, s, s, cu, cu, q, k, v, torch.bfloat16,
            FusedAttnBackend["F16_arbitrary_seqlen"], None,
            attn_bias_type="no_bias", attn_mask_type="causal",
            softmax_type="vanilla", qkv_layout="bshd_bshd_bshd",
        )
        qq = q.view(bs, s, nh, hd).permute(0, 2, 1, 3).float()
        kk = k.view(bs, s, nh, hd).permute(0, 2, 1, 3).float()
        vv = v.view(bs, s, nh, hd).permute(0, 2, 1, 3).float()
        scl = qq @ kk.transpose(-1, -2) / (hd ** 0.5)
        mask = torch.triu(torch.ones(s, s, device="cuda", dtype=torch.bool), 1)
        scl = scl.masked_fill(mask, float("-inf"))
        refa = torch.softmax(scl, -1) @ vv
        refa = refa.permute(0, 2, 1, 3).reshape(bs * s, nh, hd)
        erra = (out.float() - refa).abs().max().item()
        print(f"fused_attn_fwd max abs err = {erra:.3e} (out abs mean = {refa.abs().mean():.3e})")
        assert erra < 0.1, "fused_attn fwd mismatch"
        print("CHECK OK\n")
        sys.exit(0)

    if not args.attn_only:
        print_sep("RMSNORM FWD  (mem-bound: floor = HBM BW)")
        print(f"{'shape':<14} {'dtype':<10} {'time(us)':>9} {'GB/s':>12} {'%BW':>7} {'AI(flop/B)':>10}")
        for shape in rms_shapes:
            for dt in dtypes:
                r = bench_rmsnorm_fwd(shape, dt, timer)
                results.append(("rmsnorm_fwd", shape, str(dt), r))
                bps = r["gb_s"] * 1e9
                ai = (2.0 * shape[0] * shape[1]) / r["gb"] / 1e9  # ~2 FLOP/elem over bytes moved
                print(f"{str(shape):<14} {str(dt):<10} {r['med_ms']*1e3:>8.1f} {r['gb_s']:>12.1f} "
                      f"{bps/H100['bandwidth']*100:>6.1f}% {ai:>10.2f}")

        print_sep("RMSNORM BWD  (mem-bound: floor = HBM BW)")
        print(f"{'shape':<14} {'dtype':<10} {'time(us)':>9} {'GB/s':>12} {'%BW':>7} {'AI(flop/B)':>10}")
        for shape in rms_shapes:
            for dt in dtypes:
                r = bench_rmsnorm_bwd(shape, dt, timer)
                results.append(("rmsnorm_bwd", shape, str(dt), r))
                bps = r["gb_s"] * 1e9
                ai = (4.0 * shape[0] * shape[1]) / r["gb"] / 1e9
                print(f"{str(shape):<14} {str(dt):<10} {r['med_ms']*1e3:>8.1f} {r['gb_s']:>12.1f} "
                      f"{bps/H100['bandwidth']*100:>6.1f}% {ai:>10.2f}")

        print_sep("RMSNORM BWD+ADD  (fused residual-add backward, mem-bound)")
        print(f"{'shape':<14} {'dtype':<10} {'time(us)':>9} {'GB/s':>12} {'%BW':>7} {'AI(flop/B)':>10}")
        for shape in rms_shapes:
            for dt in dtypes:
                r = bench_rmsnorm_bwd_add(shape, dt, timer)
                results.append(("rmsnorm_bwd_add", shape, str(dt), r))
                bps = r["gb_s"] * 1e9
                ai = (4.0 * shape[0] * shape[1]) / r["gb"] / 1e9
                print(f"{str(shape):<14} {str(dt):<10} {r['med_ms']*1e3:>8.1f} {r['gb_s']:>12.1f} "
                      f"{bps/H100['bandwidth']*100:>6.1f}% {ai:>10.2f}")

    if not args.rmsnorm_only:
        print_sep("FUSED ATTENTION FWD  (compute-bound: floor = tensor-core peak)")
        print(f"{'shape':<22} {'dtype':<10} {'time(us)':>9} {'TFLOPS':>10} {'%TC':>6} {'GB/s':>12} {'AI(flop/B)':>10}")
        for shape in attn_shapes:
            for dt in attn_dtypes:
                bs, s, nh, hd = shape
                r = bench_fused_attn(shape, dt, timer, do_backward=False)
                results.append(("fused_attn_fwd", shape, str(dt), r))
                flops = 4 * bs * s * nh * s * hd
                ai = flops / (r["gb"] * 1e9)
                print(f"{str(shape):<22} {str(dt):<10} {r['med_ms']*1e3:>8.1f} {r['tflops']:>10.1f} "
                      f"{r['tflops']/H100['fp16_tflops']*100:>5.1f}% {r['gb_s']:>12.1f} {ai:>10.1f}")

        print_sep("FUSED ATTENTION BWD  (compute-bound: floor = tensor-core peak)")
        print(f"{'shape':<22} {'dtype':<10} {'time(us)':>9} {'TFLOPS':>10} {'%TC':>6} {'GB/s':>12} {'AI(flop/B)':>10}")
        for shape in attn_shapes:
            for dt in attn_dtypes:
                bs, s, nh, hd = shape
                r = bench_fused_attn(shape, dt, timer, do_backward=True)
                results.append(("fused_attn_bwd", shape, str(dt), r))
                flops = 8 * bs * s * nh * s * hd
                ai = flops / (r["gb"] * 1e9)
                print(f"{str(shape):<22} {str(dt):<10} {r['med_ms']*1e3:>8.1f} {r['tflops']:>10.1f} "
                      f"{r['tflops']/H100['fp16_tflops']*100:>5.1f}% {r['gb_s']:>12.1f} {ai:>10.1f}")

    # Roofline reference summary
    print_sep("H100 ROOFLINE REFERENCE")
    print(f"FP16/BF16 tensor-core peak : {H100['fp16_tflops']:>10.1f} TFLOPS (dense, no sparsity)")
    print(f"FP32 (CUDA-core FMA) peak  : {H100['fp32_tflops']:>10.1f} TFLOPS")
    print(f"HBM3 memory bandwidth      : {H100['bandwidth']/1e12:>10.2f} TB/s")
    for lbl, peak in [("FP16/BF16", H100['fp16_tflops']), ("FP32", H100['fp32_tflops'])]:
        ridge = peak * 1e12 / H100['bandwidth']
        print(f"Roofline ridge ({lbl})     : AI = {ridge:>8.1f} FLOP/byte "
              f"(below: mem-bound slope {H100['bandwidth']/1e9:.0f} GB/s/FLOP/byte; above: flat {peak:.0f} TFLOPS)")
    print(f"Measured launch overhead   : ~{launch_ovh*1e3:.1f} us/call (subtract for small shapes)")
    print("Note: rmsnorm* are memory-bound (AI ~2-4 FLOP/byte, far below ridge);")
    print("      fused_attn* cross the ridge as seqlen grows -> compute-bound.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kernel", "shape", "dtype", "mean_ms", "med_ms",
                        "gb_s", "gb", "tflops"])
            for kernel, shape, dt, r in results:
                w.writerow([kernel, shape, dt, r.get("mean_ms"), r.get("med_ms"),
                            r.get("gb_s"), r.get("gb"), r.get("tflops")])
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
