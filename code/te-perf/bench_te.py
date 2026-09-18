# Copyright (c) 2026. All rights reserved.
#
# Benchmark TransformerEngine kernels:
#   - te.rmsnorm (rmsnorm_fwd)
#   - te.rmsnorm_bwd
#   - te.fused_attn_fwd / te.fused_attn_bwd
#
# Timing: reports PURE DEVICE kernel time (CUPTI / torch.profiler, launch
# overhead excluded) as the primary metric, plus wall-clock CUDA-event median
# (launch included) for reference. Achieved TFLOPS / GB/s are derived from the
# pure device time. Compared against hard-coded roofline numbers for
# NVIDIA H100 SXM (132 SM, 80GB HBM3).
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
from torch.profiler import ProfilerActivity, profile

# --------------------------------------------------------------------------- #
# H100 SXM roofline constants (hard-coded, device-agnostic reference).
# --------------------------------------------------------------------------- #
#   132 SMs @ 1.980 GHz
#   FP16/BF16 tensor-core dense: ~989.4 TFLOPS (no sparsity)
#   FP8 tensor-core dense:       ~1978.8 TFLOPS
#   FP16/BF16 with 2:4 sparsity: ~1978.8 TFLOPS
#   FP32 (CUDA core FMA):       ~66.9  TFLOPS
#   HBM3 bandwidth:             ~3.35  TB/s (3.35e12 B/s)
H100 = {
    "fp16_tflops": 989.4,
    "bf16_tflops": 989.4,
    "fp8_tflops": 1978.8,
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
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
except Exception as e:  # pragma: no cover - import at bench time
    print(f"[FATAL] unable to import transformer_engine: {e}", file=sys.stderr)
    sys.exit(1)

from transformer_engine_torch import rmsnorm_fwd, rmsnorm_bwd, rmsnorm_bwd_add  # noqa: E402


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
class KernelTimer:
    """Two timers with warmup + repeat:

    - ``time`` returns wall-clock CUDA-event latency (ms), which **includes** the
      host-side launch/dispatch gap (that gap is exposed because the GPU is idle
      at the start of each timed iteration).  Useful as an end-to-end number, but
      for small kernels it is dominated by launch overhead.
    - ``device_time`` returns the pure on-device kernel execution time (ms) via
      CUPTI (``torch.profiler``), which **excludes** launch overhead.  This is the
      number to use for bandwidth / FLOPs accounting.  Note launch overhead is
      *not* a universal constant (it differs per op: deeper Python/C++ dispatch =
      larger gap), so subtracting a fixed value from the wall-clock number is
      inaccurate.
    """

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

    def device_time(self, fn, *args, **kwargs):
        """Mean pure device kernel time (ms) over ``repeat`` calls, launch excluded.

        Sums the CUPTI device duration of all kernels (and device memsets) launched
        by ``fn`` in the profiling window, then divides by the number of calls.
        """
        for _ in range(self.warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(self.repeat):
                fn(*args, **kwargs)
            torch.cuda.synchronize()
        total_us = sum(evt.device_time_total for evt in prof.key_averages())
        return total_us / self.repeat / 1e3


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

    ms = timer.device_time(fn)  # pure device time (launch excluded)
    # bytes moved: read x + read w + write y (+ rsigma, negligible)
    gb = (bytes_of(x) + bytes_of(w) * 1 + bytes_of(x)) / 1e9
    gb_s = gb / (ms / 1e3)
    return {"kernel_ms": ms, "gb_s": gb_s, "gb": gb}


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

    ms = timer.device_time(fn)  # pure device time (launch excluded)
    # read dy + read x + read rsigma + read w, write dx + write dw
    gb = (bytes_of(dy) + bytes_of(x) + rows * 4 + bytes_of(w) + bytes_of(x) + bytes_of(w)) / 1e9
    gb_s = gb / (ms / 1e3)
    return {"kernel_ms": ms, "gb_s": gb_s, "gb": gb}


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

    ms = timer.device_time(fn)  # pure device time (launch excluded)
    # read dy + read x + read add + read rsigma + read w, write dx + write dw
    gb = (bytes_of(dy) + bytes_of(x) + bytes_of(add) + rows * 4
          + bytes_of(w) + bytes_of(x) + bytes_of(w)) / 1e9
    gb_s = gb / (ms / 1e3)
    return {"kernel_ms": ms, "gb_s": gb_s, "gb": gb}


# --------------------------------------------------------------------------- #
# fused attention fwd / bwd  (training, causal, dropped 0)
# --------------------------------------------------------------------------- #
def attn_dims(shape):
    """shape = (batch, seqlen, num_heads, qk_head_dim[, v_head_dim]).

    The 4-tuple form is plain MHA (qk == v head dim). The 5-tuple form carries a
    separate v head dim for MLA (e.g. Kimi-K2.6 qk=192, v=128); TE fused_attn
    accepts q/k and v with different last dims.
    """
    bs, seqlen, num_heads, qk_dim = shape[0], shape[1], shape[2], shape[3]
    v_dim = shape[4] if len(shape) > 4 else qk_dim
    return bs, seqlen, num_heads, qk_dim, v_dim


def bench_fused_attn(shape, dtype, timer, do_backward=False):
    bs, seqlen, num_heads, qk_dim, v_dim = attn_dims(shape)

    q = torch.randn(bs * seqlen, num_heads, qk_dim, device="cuda", dtype=dtype)
    k = torch.randn(bs * seqlen, num_heads, qk_dim, device="cuda", dtype=dtype)
    v = torch.randn(bs * seqlen, num_heads, v_dim, device="cuda", dtype=dtype)

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

        ms = timer.device_time(fn_fwd)  # pure device time (launch excluded)
        # q + k + v read, o write (approx; s/m intermediate ignored)
        gb = (bytes_of(q) + bytes_of(k) + bytes_of(v) + bytes_of(out)) / 1e9
        gb_s = gb / (ms / 1e3)
        # FLOPs: QK^T (2*b*s*h*s*qk_dim) + PV (2*b*s*h*s*v_dim)
        flops = 2 * bs * seqlen * num_heads * seqlen * (qk_dim + v_dim)
        tflops = flops / (ms / 1e3) / 1e12
        return {"kernel_ms": ms, "gb_s": gb_s, "gb": gb, "tflops": tflops}

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

    ms = timer.device_time(fn_bwd)  # pure device time (launch excluded)
    # read q,k,v,o,d_o; write dq,dk,dv
    gb = (bytes_of(q) + bytes_of(k) + bytes_of(v) + bytes_of(out) + bytes_of(d_out)
          + bytes_of(q) + bytes_of(k) + bytes_of(v)) / 1e9
    gb_s = gb / (ms / 1e3)
    # FLOPs: backward ~ 2x forward = 4*b*s*h*s*(qk_dim + v_dim)
    flops = 4 * bs * seqlen * num_heads * seqlen * (qk_dim + v_dim)
    tflops = flops / (ms / 1e3) / 1e12
    return {"kernel_ms": ms, "gb_s": gb_s, "gb": gb, "tflops": tflops}


def _fp8_quantizer(fp8_dtype):
    """Rowwise-only FP8 quantizer, as used internally by TE's fused attention."""
    return Float8Quantizer(
        scale=torch.ones(1, device="cuda"),
        amax=torch.zeros(1, device="cuda"),
        fp8_dtype=fp8_dtype,
        rowwise=True,
        columnwise=False,
    )


def bench_fused_attn_fp8(shape, timer, do_backward=False):
    """FP8 fused attention (E4M3 for fwd QKV/S/O, E5M2 for bwd dO/dP/dQKV).

    Q/K/V (and dO) are quantized to FP8 *outside* the timed region, so the
    measured device time is the pure FP8 fused-attention kernel; per-call input
    quantization is not included (matching the high-precision path's scope).
    """
    bs, seqlen, num_heads, qk_dim, v_dim = attn_dims(shape)
    nominal = torch.bfloat16
    e4m3 = tex.DType.kFloat8E4M3
    e5m2 = tex.DType.kFloat8E5M2

    q = torch.randn(bs * seqlen, num_heads, qk_dim, device="cuda", dtype=nominal)
    k = torch.randn(bs * seqlen, num_heads, qk_dim, device="cuda", dtype=nominal)
    v = torch.randn(bs * seqlen, num_heads, v_dim, device="cuda", dtype=nominal)

    cu_seqlens = torch.arange(0, (bs + 1) * seqlen, seqlen, dtype=torch.int32, device="cuda")
    max_seqlen = seqlen

    qkv_q = _fp8_quantizer(e4m3)
    s_q = _fp8_quantizer(e4m3)
    o_q = _fp8_quantizer(e4m3)
    do_q = _fp8_quantizer(e5m2)
    dp_q = _fp8_quantizer(e5m2)
    dqkv_q = _fp8_quantizer(e5m2)

    q8, k8, v8 = qkv_q(q), qkv_q(k), qkv_q(v)
    backend = FusedAttnBackend["FP8"]
    attn_bias_type = "no_bias"
    attn_mask_type = "causal"
    softmax_type = "vanilla"

    out, aux_ctx, *_ = fused_attn_fwd(
        True, max_seqlen, max_seqlen, cu_seqlens, cu_seqlens,
        q8, k8, v8, nominal, backend, None,
        s_quantizer=s_q, o_quantizer=o_q,
        attn_bias_type=attn_bias_type, attn_mask_type=attn_mask_type,
        softmax_type=softmax_type, qkv_layout="bshd_bshd_bshd",
    )

    if not do_backward:
        def fn_fwd():
            o, _, *_ = fused_attn_fwd(
                True, max_seqlen, max_seqlen, cu_seqlens, cu_seqlens,
                q8, k8, v8, nominal, backend, None,
                s_quantizer=s_q, o_quantizer=o_q,
                attn_bias_type=attn_bias_type, attn_mask_type=attn_mask_type,
                softmax_type=softmax_type, qkv_layout="bshd_bshd_bshd",
            )
            return o

        ms = timer.device_time(fn_fwd)
        # fp8 = 1 byte/element; q + k + v read, o write
        gb = (q.numel() + k.numel() + v.numel() + out.numel()) / 1e9
        gb_s = gb / (ms / 1e3)
        flops = 2 * bs * seqlen * num_heads * seqlen * (qk_dim + v_dim)
        tflops = flops / (ms / 1e3) / 1e12
        return {"kernel_ms": ms, "gb_s": gb_s, "gb": gb, "tflops": tflops}

    d_out = torch.randn(bs * seqlen, num_heads, v_dim, device="cuda", dtype=nominal)
    d_out8 = do_q(d_out)

    def fn_bwd():
        dqkv = fused_attn_bwd(
            max_seqlen, max_seqlen, cu_seqlens, cu_seqlens,
            q8, k8, v8, out, d_out8, nominal, d_out8._fp8_dtype,
            list(aux_ctx), backend,
            qkv_layout="bshd_bshd_bshd",
            s_quantizer=s_q, dp_quantizer=dp_q, dqkv_quantizer=dqkv_q,
            attn_bias_type=attn_bias_type, attn_mask_type=attn_mask_type,
            softmax_type=softmax_type,
        )
        return dqkv

    ms = timer.device_time(fn_bwd)
    # fp8 = 1 byte/element; read q,k,v,o,d_o; write dq,dk,dv
    gb = (q.numel() + k.numel() + v.numel() + out.numel() + d_out.numel()
          + q.numel() + k.numel() + v.numel()) / 1e9
    gb_s = gb / (ms / 1e3)
    flops = 4 * bs * seqlen * num_heads * seqlen * (qk_dim + v_dim)
    tflops = flops / (ms / 1e3) / 1e12
    return {"kernel_ms": ms, "gb_s": gb_s, "gb": gb, "tflops": tflops}


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
        (8192, 1024),
        (8192, 2048),
        (8192, 4096),
        (16384, 2048),
        (32768, 2048),
        (4096, 7168),   # kimi2.6: hidden 7168
        (8192, 7168),   # kimi2.6
        (16384, 4096),  # dsv4
        (16384, 7168),  # kimi2.6
    ]
    dtypes = [torch.float32, torch.bfloat16, torch.float16]

    # attention shapes: (batch, seqlen, num_heads, qk_head_dim[, v_head_dim])
    # The 4-tuple form is plain MHA; the 5-tuple form is MLA (qk != v).
    attn_shapes = [
        (1, 512, 16, 128),
        (1, 1024, 16, 128),
        (1, 2048, 16, 128),
        (2, 2048, 16, 128),
        (4, 2048, 16, 128),
        (8, 2048, 16, 128),
        (1, 4096, 16, 128),
        (2, 4096, 16, 128),
        (1, 1024, 32, 128),
        (4, 1024, 32, 128),
        (8, 1024, 32, 128),
        (4, 8192, 16, 128),
        # production models (batch=1, seq=4096):
        # NOTE: dsv4/dsv4.1 MLA core attention uses head_dim=512 (qk=448+64, v=512),
        #       which TE fused_attn does NOT support (H100 max 256); those models run
        #       their sparse attention with custom CSA/DSA kernels. What TE *can* run
        #       for them is the DSA indexer (head_dim=128).
        (1, 4096, 64, 192, 128),  # Kimi-K2.6 MLA: qk=nope128+rope64=192, v=128, 64 heads
        (1, 4096, 64, 128, 128),  # dsv4 DSA indexer: 64 heads, head_dim=128
        (1, 4096, 32, 128, 128),  # dsv4.1 DSA indexer: 32 heads, head_dim=128
    ]
    attn_dtypes = [torch.bfloat16, torch.float16]
    # subset also run in FP8 (E4M3 fwd / E5M2 bwd); label "fp8" in the CSV
    attn_fp8_shapes = [
        (1, 1024, 32, 128),
    ]

    results = []

    # reference: host-side launch/dispatch overhead of a trivial kernel
    empty = torch.zeros(1, device="cuda")
    def noop():
        empty.fill_(1.0)
    _, launch_ms, _ = timer.time(noop)
    noop_kernel_ms = timer.device_time(noop)
    print_sep("TIMING METHOD")
    print("time columns below = CUPTI pure device kernel time (launch overhead EXCLUDED)")
    print(f"trivial fill_ round-trip: wall={launch_ms*1e3:.1f} us, device={noop_kernel_ms*1e3:.2f} us")
    print(f"-> host launch/dispatch overhead ~= {(launch_ms - noop_kernel_ms)*1e3:.1f} us for a trivial op")
    print("launch overhead is NOT a universal constant (deeper host dispatch => larger gap),")
    print("so wall-clock time cannot be corrected by subtracting a single fixed value.")

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
                print(f"{str(shape):<14} {str(dt):<10} {r['kernel_ms']*1e3:>8.1f} {r['gb_s']:>12.1f} "
                      f"{bps/H100['bandwidth']*100:>6.1f}% {ai:>10.2f}")

        print_sep("RMSNORM BWD  (mem-bound: floor = HBM BW)")
        print(f"{'shape':<14} {'dtype':<10} {'time(us)':>9} {'GB/s':>12} {'%BW':>7} {'AI(flop/B)':>10}")
        for shape in rms_shapes:
            for dt in dtypes:
                r = bench_rmsnorm_bwd(shape, dt, timer)
                results.append(("rmsnorm_bwd", shape, str(dt), r))
                bps = r["gb_s"] * 1e9
                ai = (4.0 * shape[0] * shape[1]) / r["gb"] / 1e9
                print(f"{str(shape):<14} {str(dt):<10} {r['kernel_ms']*1e3:>8.1f} {r['gb_s']:>12.1f} "
                      f"{bps/H100['bandwidth']*100:>6.1f}% {ai:>10.2f}")

        print_sep("RMSNORM BWD+ADD  (fused residual-add backward, mem-bound)")
        print(f"{'shape':<14} {'dtype':<10} {'time(us)':>9} {'GB/s':>12} {'%BW':>7} {'AI(flop/B)':>10}")
        for shape in rms_shapes:
            for dt in dtypes:
                r = bench_rmsnorm_bwd_add(shape, dt, timer)
                results.append(("rmsnorm_bwd_add", shape, str(dt), r))
                bps = r["gb_s"] * 1e9
                ai = (4.0 * shape[0] * shape[1]) / r["gb"] / 1e9
                print(f"{str(shape):<14} {str(dt):<10} {r['kernel_ms']*1e3:>8.1f} {r['gb_s']:>12.1f} "
                      f"{bps/H100['bandwidth']*100:>6.1f}% {ai:>10.2f}")

    if not args.rmsnorm_only:
        print_sep("FUSED ATTENTION FWD  (compute-bound: floor = tensor-core peak)")
        print(f"{'shape':<22} {'dtype':<10} {'time(us)':>9} {'TFLOPS':>10} {'%TC':>6} {'GB/s':>12} {'AI(flop/B)':>10}")
        for shape in attn_shapes:
            for dt in attn_dtypes:
                bs, s, nh, qk_dim, v_dim = attn_dims(shape)
                r = bench_fused_attn(shape, dt, timer, do_backward=False)
                results.append(("fused_attn_fwd", shape, str(dt), r))
                flops = 2 * bs * s * nh * s * (qk_dim + v_dim)
                ai = flops / (r["gb"] * 1e9)
                print(f"{str(shape):<22} {str(dt):<10} {r['kernel_ms']*1e3:>8.1f} {r['tflops']:>10.1f} "
                      f"{r['tflops']/H100['fp16_tflops']*100:>5.1f}% {r['gb_s']:>12.1f} {ai:>10.1f}")
        for shape in attn_fp8_shapes:
            bs, s, nh, qk_dim, v_dim = attn_dims(shape)
            r = bench_fused_attn_fp8(shape, timer, do_backward=False)
            results.append(("fused_attn_fwd", shape, "fp8", r))
            flops = 2 * bs * s * nh * s * (qk_dim + v_dim)
            ai = flops / (r["gb"] * 1e9)
            print(f"{str(shape):<22} {'fp8':<10} {r['kernel_ms']*1e3:>8.1f} {r['tflops']:>10.1f} "
                  f"{r['tflops']/H100['fp8_tflops']*100:>5.1f}% {r['gb_s']:>12.1f} {ai:>10.1f}")

        print_sep("FUSED ATTENTION BWD  (compute-bound: floor = tensor-core peak)")
        print(f"{'shape':<22} {'dtype':<10} {'time(us)':>9} {'TFLOPS':>10} {'%TC':>6} {'GB/s':>12} {'AI(flop/B)':>10}")
        for shape in attn_shapes:
            for dt in attn_dtypes:
                bs, s, nh, qk_dim, v_dim = attn_dims(shape)
                r = bench_fused_attn(shape, dt, timer, do_backward=True)
                results.append(("fused_attn_bwd", shape, str(dt), r))
                flops = 4 * bs * s * nh * s * (qk_dim + v_dim)
                ai = flops / (r["gb"] * 1e9)
                print(f"{str(shape):<22} {str(dt):<10} {r['kernel_ms']*1e3:>8.1f} {r['tflops']:>10.1f} "
                      f"{r['tflops']/H100['fp16_tflops']*100:>5.1f}% {r['gb_s']:>12.1f} {ai:>10.1f}")
        for shape in attn_fp8_shapes:
            bs, s, nh, qk_dim, v_dim = attn_dims(shape)
            r = bench_fused_attn_fp8(shape, timer, do_backward=True)
            results.append(("fused_attn_bwd", shape, "fp8", r))
            flops = 4 * bs * s * nh * s * (qk_dim + v_dim)
            ai = flops / (r["gb"] * 1e9)
            print(f"{str(shape):<22} {'fp8':<10} {r['kernel_ms']*1e3:>8.1f} {r['tflops']:>10.1f} "
                  f"{r['tflops']/H100['fp8_tflops']*100:>5.1f}% {r['gb_s']:>12.1f} {ai:>10.1f}")

    # Roofline reference summary
    print_sep("H100 ROOFLINE REFERENCE")
    print(f"FP16/BF16 tensor-core peak : {H100['fp16_tflops']:>10.1f} TFLOPS (dense, no sparsity)")
    print(f"FP8 tensor-core peak       : {H100['fp8_tflops']:>10.1f} TFLOPS (dense, no sparsity)")
    print(f"FP32 (CUDA-core FMA) peak  : {H100['fp32_tflops']:>10.1f} TFLOPS")
    print(f"HBM3 memory bandwidth      : {H100['bandwidth']/1e12:>10.2f} TB/s")
    for lbl, peak in [("FP16/BF16", H100['fp16_tflops']), ("FP8", H100['fp8_tflops']),
                      ("FP32", H100['fp32_tflops'])]:
        ridge = peak * 1e12 / H100['bandwidth']
        print(f"Roofline ridge ({lbl})     : AI = {ridge:>8.1f} FLOP/byte "
              f"(below: mem-bound slope {H100['bandwidth']/1e9:.0f} GB/s/FLOP/byte; above: flat {peak:.0f} TFLOPS)")
    print(f"Trivial-kernel round-trip  : ~{launch_ovh*1e3:.1f} us/call wall (host dispatch, not subtracted)")
    print("Note: rmsnorm* are memory-bound (AI ~2-4 FLOP/byte, far below ridge);")
    print("      fused_attn* cross the ridge as seqlen grows -> compute-bound.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            # kernel_ms = CUPTI pure device kernel time (launch overhead excluded)
            w.writerow(["kernel", "shape", "dtype", "kernel_ms",
                        "gb_s", "gb", "tflops"])
            for kernel, shape, dt, r in results:
                w.writerow([kernel, shape, dt, r.get("kernel_ms"),
                            r.get("gb_s"), r.get("gb"), r.get("tflops")])
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
