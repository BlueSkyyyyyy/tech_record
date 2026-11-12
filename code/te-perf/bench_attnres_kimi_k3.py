#!/usr/bin/env python3
# Copyright (c) 2026. All rights reserved.
#
# Benchmark FLA's **AttnRes** operator (`fla.ops.attnres`) on the *actual Kimi K3* shapes,
# plus correctness checks against an fp64 reference and a per-iteration cost model for K3.
#
# What AttnRes is
# ---------------
# Kimi K3 replaces the usual PreNorm `h = h + sublayer(h)` residual with a per-token
# softmax attention over *depth* sources (Attention Residuals, arXiv:2603.15031):
#
#     h = sum_l p_l * V_l,     p = softmax_l( <RMSNorm(V_l), q_layer> )
#
# `V` = [b_0 .. b_{k-1}, partial]: b_i are the summed outputs of completed depth blocks
# and `partial` is the running intra-block sum. K3 config (`text_config`):
#   hidden_size D = 7168, 93 layers, attn_res_block_size = 12 layers (= 24 sublayers)
#   => 8 completed blocks max, so a sublayer sees L = 2..9 sources (L=1 never occurs
#      after the first block boundary; the very first aggregation is over 2 sources).
# So this is a **bandwidth-bound** op whose cost scales with L: it streams
# `L * num_tokens * 7168` bf16 elements per call, twice per transformer layer.
#
# References used for the shapes / call convention
# ------------------------------------------------
#   * ~/models/Kimi-K3/config.json                              (D, layers, block size)
#   * .../modeling_kimi_linear.py:_apply_attn_res               (exact math + L schedule)
#   * Megatron-LM branch `to-main/attnres-6840`:
#       megatron/core/transformer/attention_residual.py         (FLA call site:
#         output_rms_weight=norm weight, scale=1.0, checkpoint_level=1, fla impl)
#       tests/.../test_kimi_k3_attnres_mtp.py                   (L=9 max, D=7168)
#   * FLA `fla/ops/attnres/{fused,naive}.py` and its gluon backend.
#
# Modes
# -----
#   --mode check     fused (Triton & Gluon) vs fp64 golden: forward + every VJP.
#   --mode sweep     the performance grid: T x L x backend, fwd / bwd / fwd+bwd,
#                    wall (CUDA events) AND pure device time (CUPTI, launch excluded).
#   --mode schedule  aggregates the measured per-L cost over K3's real 93-layer schedule
#                    (186 AttnRes calls per iteration) -> per-iteration overhead.
#   --mode extras    output-RMSNorm fusion, checkpoint_level 0 vs 1, L2 compile bucket.
#   --mode all       check + sweep + schedule + extras (default).
#
# Timing conventions (same as bench_kda_kimi_k3.py / bench_mla_kimi_k3.py)
#   wall_ms   = CUDA-event mean over `repeat` calls, launch included.
#   dev_ms    = PURE device kernel time (CUPTI via torch.profiler), launch excluded.
#   bwd       = backward ONLY: the graph is built once outside the timed loop and only
#               autograd.grad(..., retain_graph=True) is timed (FLA recomputes o_pre
#               inside backward for checkpoint_level=1, that recompute is included).
#   step      = fwd+bwd with a fresh graph each call.
#
# How to run (inside the lab container, on an IDLE GPU; FLA main is 0.6.0 while
# the container's installed fla is 0.5.2, so point FLA_SRC at the checkout):
#
#   export FLA_SRC=~/proj/flash-linear-attention
#   # FLA main imports `fla.models`, which needs transformers>=5 (PreTrainedConfig);
#   # the script aliases that symbol when the container has transformers 4.x.
#   # tilelang's TVM needs libz3 which lives in the Megatron-Bridge venv:
#   export LD_LIBRARY_PATH=~/proj/Megatron-Bridge/.venv/lib/python3.12/site-packages/z3/lib:$LD_LIBRARY_PATH
#   CUDA_VISIBLE_DEVICES=7 python bench_attnres_kimi_k3.py --mode all
#
# NOTE: local absolute paths are intentionally avoided in this file (FLA_SRC is env-driven).

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

# H100 SXM5 (the card used for the numbers in results_attnres_kimi_k3.md).
H100 = {"bandwidth": 3.35e12, "bf16_tflops": 989.4}

# Kimi K3 text_config (HF config.json + Megatron recipe attn_res_block_layers=24 sublayers).
K3 = {
    "hidden": 7168,
    "num_layers": 93,
    "attn_res_block_layers": 12,   # HF block size in *layers* (= 24 sublayers)
    "num_heads": 96,
}

_FUSED = None
_NAIVE = None


# --------------------------------------------------------------------------- #
# FLA import bootstrap
# --------------------------------------------------------------------------- #
def ensure_fla(fla_src: str | None = None):
    """Import FLA after putting the checkout first on sys.path.

    FLA main (0.6.0) imports ``fla.models`` at package import time, which expects
    transformers>=5 (``PreTrainedConfig``). Kernel containers often ship 4.x, where
    the class is still ``PretrainedConfig``. AttnRes never touches transformers, so
    aliasing the symbol is enough to import the op.
    """
    global _FUSED, _NAIVE
    if _FUSED is not None:
        return _FUSED, _NAIVE

    src = fla_src or os.environ.get("FLA_SRC")
    if src:
        sys.path.insert(0, src)
    try:
        import transformers.configuration_utils as cu
        if not hasattr(cu, "PreTrainedConfig"):
            cu.PreTrainedConfig = cu.PretrainedConfig
    except Exception:  # noqa: BLE001 - transformers is optional for the op
        pass

    import fla
    from fla.ops.attnres import fused_attnres, naive_attnres

    print(f"fla {getattr(fla, '__version__', '?')} @ {fla.__file__}")
    _FUSED, _NAIVE = fused_attnres, naive_attnres
    return _FUSED, _NAIVE


def set_backend(name: str) -> None:
    """Requested backend: 'triton' (default function) or 'gluon' (FLA_ATTNRES_GLUON=1).

    FLA's dispatch re-reads the env var on every call, so it can be toggled per config.
    """
    os.environ["FLA_ATTNRES_GLUON"] = "1" if name == "gluon" else "0"


# --------------------------------------------------------------------------- #
# Timer
# --------------------------------------------------------------------------- #
class KernelTimer:
    """Wall (CUDA events) and pure device (CUPTI) timers."""

    def __init__(self, warmup: int = 5, repeat: int = 20):
        self.warmup = warmup
        self.repeat = repeat
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def wall_ms(self, fn) -> float:
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(self.repeat):
            self.start.record()
            fn()
            self.end.record()
            torch.cuda.synchronize()
            ts.append(self.start.elapsed_time(self.end))
        return sum(ts) / len(ts)

    def device_ms(self, fn) -> tuple[float, dict]:
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(self.repeat):
                fn()
            torch.cuda.synchronize()
        per_kernel = {
            e.key: e.device_time_total / self.repeat / 1e3
            for e in prof.key_averages() if e.device_time_total
        }
        return sum(per_kernel.values()), per_kernel


def top_kernels(per_kernel: dict, k: int = 2) -> str:
    items = sorted(per_kernel.items(), key=lambda kv: -kv[1])[:k]
    return " ".join(f"{name}={ms * 1e3:.0f}us" for name, ms in items)


# --------------------------------------------------------------------------- #
# Shapes / bytes
# --------------------------------------------------------------------------- #
def k3_sublayer_schedule(num_layers: int = 93, block_layers: int = 12):
    """(layer_idx, sublayer, L) for every AttnRes call, mirroring K3's forward.

    ``modeling_kimi_linear._forward_attn_residual`` appends the running partial sum to
    ``block_residual`` when ``layer_idx % block_size == 0``; every aggregation then sees
    ``#appended_blocks + 1`` sources (the +1 is the live partial sum). The attention
    aggregation of a block-start layer runs before the append and is skipped for layer 0
    (Megatron ``AttentionResidual._aggregate`` short-circuits the one-source case as an
    identity). The final output head (``_apply_output_attn_res`` /
    ``final_attn_res``) aggregates all blocks plus the trailing partial, so it is one
    extra call with the largest L.
    """
    out = []
    blocks = 0
    for i in range(num_layers):
        if i % block_layers == 0:
            if blocks:
                out.append((i, "attn", blocks + 1))
            blocks += 1
            out.append((i, "mlp", blocks + 1))
        else:
            out.append((i, "attn", blocks + 1))
            out.append((i, "mlp", blocks + 1))
    # final output aggregation before the last RMSNorm
    out.append((num_layers, "out", blocks + 1))
    return out


def bytes_fwd(N: int, D: int, L: int, es: int, onorm: bool = False, ckpt: int = 1) -> int:
    """Read L sources + write the mixed residual (+ [L,N] fp32 rstd/logit, [N] lse).

    ckpt=0 additionally writes the pre-norm mixed residual `o_pre` for backward.
    """
    b = (L + 1) * N * D * es + 2 * L * N * 4 + N * 4
    if ckpt == 0:
        b += N * D * es
    return b


def bytes_bwd(N: int, D: int, L: int, es: int, onorm: bool = True, ckpt: int = 1) -> int:
    """Backward traffic.

    * sources are read twice with ckpt=1 (once to rebuild o_pre, once for dV), once with ckpt=0;
    * do is read once, dV written once per source;
    * FLA stages `dqw` as an [N, D] **fp32** partial (written by bwd_dv, read by bwd_dqdw) to
      avoid atomics, and one more fp32 [N, D] `dow_partial` when the output RMSNorm is fused.
      Those two scratch buffers are real HBM traffic and are counted here.
    """
    reads = L + (L if ckpt == 1 else 0) + 1
    b = (reads + L) * N * D * es
    b += 2 * N * D * 4                       # dqw: write + read, fp32
    if onorm:
        b += 2 * N * D * 4                   # dow_partial: write + read, fp32
    return b


# --------------------------------------------------------------------------- #
# Inputs / call helpers
# --------------------------------------------------------------------------- #
def make_inputs(T: int, D: int, L: int, dtype=torch.bfloat16, B: int = 1, seed: int = 0):
    torch.manual_seed(seed)
    dev = "cuda"
    # Megatron keeps the AttnRes pseudo-query / key-norm weight in fp32 (mark_keep_in_fp32).
    q = torch.randn(D, device=dev, dtype=torch.float32, requires_grad=True)
    w = torch.ones(D, device=dev, dtype=torch.float32, requires_grad=True)
    ow = torch.ones(D, device=dev, dtype=torch.float32, requires_grad=True)
    res = [(torch.randn(B, T, D, device=dev, dtype=dtype) * 0.02).requires_grad_(True)
           for _ in range(L)]
    return q, w, ow, res


def call_fused(q, w, ow, res, onorm: bool = True, ckpt: int = 1):
    return _FUSED(
        q, res, w,
        output_rms_weight=ow if onorm else None,
        rms_eps=1e-6,
        scale=1.0,
        checkpoint_level=ckpt,
    )


def call_naive(q, w, ow, res, onorm: bool = True):
    return _NAIVE(
        q, res, w,
        output_rms_weight=ow if onorm else None,
        rms_eps=1e-6,
        scale=1.0,
    )


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def measure_config(backend: str, T: int, D: int, L: int, onorm: bool, ckpt: int,
                   timer: KernelTimer, dtype=torch.bfloat16) -> dict:
    set_backend(backend if backend in ("triton", "gluon") else "triton")
    q, w, ow, res = make_inputs(T, D, L, dtype)
    do = torch.randn_like(res[0])
    inputs = [*res, q, w] + ([ow] if onorm else [])

    if backend == "naive":
        fwd = lambda: call_naive(q, w, ow, res, onorm)  # noqa: E731
    else:
        fwd = lambda: call_fused(q, w, ow, res, onorm, ckpt)  # noqa: E731

    row = dict(backend=backend, dtype=str(dtype).split(".")[-1], T=T, D=D, L=L,
               onorm=int(onorm), ckpt=ckpt if backend != "naive" else -1)

    try:
        o = fwd()
        torch.cuda.synchronize()
        dev, kernels = timer.device_ms(fwd)
        row.update(fwd_dev=dev, fwd_wall=timer.wall_ms(fwd),
                   fwd_gbps=(float('nan') if backend == 'naive' else
                             bytes_fwd(T, D, L, res[0].element_size(), onorm, ckpt) / (dev / 1e3) / 1e9),
                   fwd_kernels=top_kernels(kernels))
    except Exception as e:  # noqa: BLE001
        row["error"] = f"fwd:{type(e).__name__}:{str(e)[:60]}"
        return row

    def bwd():
        torch.autograd.grad(o, inputs, do, retain_graph=True, allow_unused=True)

    try:
        bwd()
        torch.cuda.synchronize()
        dev, kernels = timer.device_ms(bwd)
        row.update(bwd_dev=dev, bwd_wall=timer.wall_ms(bwd),
                   bwd_gbps=(float('nan') if backend == 'naive' else
                             bytes_bwd(T, D, L, res[0].element_size(), onorm, ckpt) / (dev / 1e3) / 1e9),
                   bwd_kernels=top_kernels(kernels))
    except Exception as e:  # noqa: BLE001
        row["error"] = f"bwd:{type(e).__name__}:{str(e)[:60]}"
        return row

    def step():
        o2 = fwd()
        torch.autograd.grad(o2, inputs, do)

    try:
        dev, _ = timer.device_ms(step)
        row.update(step_dev=dev, step_wall=timer.wall_ms(step))
    except Exception as e:  # noqa: BLE001
        row["error"] = f"step:{type(e).__name__}:{str(e)[:60]}"
        return row

    try:
        row["peak_mb"] = peak_mem_mb(step)
    except Exception:  # noqa: BLE001
        row["peak_mb"] = float("nan")
    return row


def peak_mem_mb(fn) -> float:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    torch.cuda.empty_cache()
    return peak


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def golden_attnres(q, res, w, ow, eps=1e-6, scale=1.0):
    """fp64 reference of `_apply_attn_res` (FLA convention: rms_weight folded into k)."""
    v = torch.stack([r.double() for r in res])                    # [L, B, T, D]
    k = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps) * w.double()
    scores = (k * (q.double() * scale)).sum(-1)                   # [L, B, T]
    probs = scores.softmax(0)
    o = (probs.unsqueeze(-1) * v).sum(0)
    if ow is not None:
        o = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + eps) * ow.double()
    return o


def _cmp(a, b) -> tuple[float, float, float]:
    a = a.flatten().double()
    b = b.flatten().double()
    cos = torch.nn.functional.cosine_similarity(a[None], b[None]).item()
    rel = (a - b).norm() / b.norm().clamp_min(1e-12)
    return cos, rel.item(), (a - b).abs().max().item()


def check(seqs=(1024,), Ls=(2, 9), backends=("triton", "gluon"), dtype=torch.bfloat16):
    D = K3["hidden"]
    print("=" * 104)
    print("Correctness: FLA fused vs fp64 golden (forward + all VJPs), Kimi-K3 D=7168, bf16 inputs")
    print("=" * 104)
    print(f"{'backend':>8} {'T':>6} {'L':>3} {'onorm':>5} | "
          f"{'cos(o)':>9} {'rel(o)':>8} | {'cos(dV)':>9} {'rel(dV)':>8} | "
          f"{'cos(dq)':>9} {'cos(dw)':>9} {'cos(dow)':>9}")
    print("-" * 104)
    rows = []
    for backend in backends:
        set_backend(backend)
        for T in seqs:
            for L in Ls:
                for onorm in (True, False):
                    q, w, ow, res = make_inputs(T, D, L, dtype,
                                                seed=0)
                    do = torch.randn_like(res[0])

                    ref_q = q.detach().double().requires_grad_(True)
                    ref_w = w.detach().double().requires_grad_(True)
                    ref_ow = ow.detach().double().requires_grad_(True)
                    ref_res = [r.detach().double().requires_grad_(True) for r in res]
                    ref_o = golden_attnres(ref_q, ref_res, ref_w, ref_ow if onorm else None)
                    ref_inputs = [*ref_res, ref_q, ref_w] + ([ref_ow] if onorm else [])
                    ref_grads = torch.autograd.grad((ref_o * do.double()).sum(), ref_inputs)

                    fwd = (lambda: call_naive(q, w, ow, res, onorm)) if backend == "naive" \
                        else (lambda: call_fused(q, w, ow, res, onorm, 1))
                    o = fwd()
                    grads = torch.autograd.grad((o * do).sum(),
                                                [*res, q, w] + ([ow] if onorm else []))

                    cos_o, rel_o, _ = _cmp(o, ref_o)
                    dvs = [_cmp(g, r) for g, r in zip(grads[:L], ref_grads[:L])]
                    cos_dv = min(c[0] for c in dvs)
                    rel_dv = max(c[1] for c in dvs)
                    cos_q, rel_q, _ = _cmp(grads[L], ref_grads[L])
                    cos_w = _cmp(grads[L + 1], ref_grads[L + 1])[0]
                    cos_ow = _cmp(grads[L + 2], ref_grads[L + 2])[0] if onorm else float('nan')
                    print(f"{backend:>8} {T:>6} {L:>3} {int(onorm):>5} | "
                          f"{cos_o:>9.7f} {rel_o:>8.2e} | {cos_dv:>9.7f} {rel_dv:>8.2e} | "
                          f"{cos_q:>9.7f} {cos_w:>9.7f} {cos_ow:>9.7f}")
                    rows.append(dict(backend=backend, T=T, L=L, onorm=int(onorm),
                                     cos_o=cos_o, rel_o=rel_o, cos_dv=cos_dv, rel_dv=rel_dv,
                                     cos_dq=cos_q, rel_dq=rel_q, cos_dw=cos_w, cos_dow=cos_ow))
    print("-" * 104)
    print("note: bf16 inputs + fp32 accumulation vs fp64; cos ~1-1e-6 and rel ~1e-3 is expected.")
    return rows


def sweep(seqs, Ls, backends, timer, rows_out, dtype=torch.bfloat16, onorm=True, ckpt=1,
          naive_max_tokens=8192):
    D = K3["hidden"]
    print("=" * 132)
    print(f"AttnRes perf, Kimi-K3 hidden D={D}, bf16, onorm={int(onorm)}, ckpt={ckpt} | "
          f"wall=events, dev=CUPTI kernel time")
    print("=" * 132)
    print(f"{'backend':>8} {'T':>6} {'L':>3} {'fwd_dev':>8} {'fwd_wall':>9} {'GB/s':>7} | "
          f"{'bwd_dev':>8} {'bwd_wall':>9} {'GB/s':>7} | {'step_dev':>9} {'peakMB':>7} | kernels")
    print("-" * 132)
    for backend in backends:
        for T in seqs:
            for L in Ls:
                if backend == "naive" and T > naive_max_tokens:
                    continue
                r = measure_config(backend, T, D, L, onorm, ckpt, timer, dtype)
                rows_out.append(r)
                if "error" in r:
                    print(f"{backend:>8} {T:>6} {L:>3} | {r['error']}")
                    continue
                print(f"{backend:>8} {T:>6} {L:>3} "
                      f"{r['fwd_dev']:>8.3f} {r['fwd_wall']:>9.3f} {r['fwd_gbps']:>7.0f} | "
                      f"{r['bwd_dev']:>8.3f} {r['bwd_wall']:>9.3f} {r['bwd_gbps']:>7.0f} | "
                      f"{r['step_dev']:>9.3f} {r.get('peak_mb', float('nan')):>7.0f} | "
                      f"{r.get('fwd_kernels', '')}")
    print("-" * 132)
    print("GB/s = analytic bytes / CUPTI time (includes the fp32 dqw / dow_partial scratch; "
          "see bytes_*()); H100 peak 3350 GB/s.")
    return rows_out


def schedule_mode(seqs, timer, dtype=torch.bfloat16, onorm=True, ckpt=1):
    """Per-iteration AttnRes cost of a real K3 layer stack, from per-L measurements."""
    D = K3["hidden"]
    sched = k3_sublayer_schedule(K3["num_layers"], K3["attn_res_block_layers"])
    counts = {}
    for _, _, L in sched:
        counts[L] = counts.get(L, 0) + 1
    print("=" * 100)
    print(f"K3 AttnRes per-iteration cost model: {K3['num_layers']} layers, block="
          f"{K3['attn_res_block_layers']} layers, {len(sched)} AttnRes calls/iteration")
    print("=" * 100)
    print(f"L distribution: " + ", ".join(f"L={L}:{c}" for L, c in sorted(counts.items())))
    print()
    print(f"{'T':>6} {'L':>3} {'calls':>5} {'fwd_us':>8} {'bwd_us':>8} {'tot_ms':>8} | "
          f"{'attnres_GB':>10} {'min_ms@3350':>11}")
    print("-" * 100)
    for T in seqs:
        per_L = {}
        for L in sorted(counts):
            set_backend("triton")
            q, w, ow, res = make_inputs(T, D, L, dtype)
            do = torch.randn_like(res[0])
            fwd = lambda: call_fused(q, w, ow, res, onorm, ckpt)  # noqa: E731
            o = fwd()
            fwd_dev, _ = timer.device_ms(fwd)
            bwd = lambda: torch.autograd.grad(o, [*res, q, w, ow], do, retain_graph=True)  # noqa: E731
            bwd_dev, _ = timer.device_ms(bwd)
            per_L[L] = (fwd_dev, bwd_dev)

        tot_fwd = sum(counts[L] * per_L[L][0] for L in counts)
        tot_bwd = sum(counts[L] * per_L[L][1] for L in counts)
        nbytes = sum(counts[L] * (bytes_fwd(T, D, L, 2, onorm, ckpt) + bytes_bwd(T, D, L, 2, onorm, ckpt))
                     for L in counts)
        print(f"{T:>6} {'':>3} {len(sched):>5} {tot_fwd * 1e3:>8.0f} {tot_bwd * 1e3:>8.0f} "
              f"{(tot_fwd + tot_bwd):>8.2f} | {nbytes / 1e9:>10.1f} "
              f"{nbytes / H100['bandwidth'] * 1e3:>11.2f}")
        for L in sorted(counts):
            f, b = per_L[L]
            print(f"{'':>6} {L:>3} {counts[L]:>5} {f * 1e3:>8.0f} {b * 1e3:>8.0f} "
                  f"{(f + b) * counts[L]:>8.2f} |")
        print("-" * 100)
    print("tot_ms = sum over the K3 schedule of per-call fwd+bwd CUPTI time at this T.")
    print("min_ms@3350 = bytes / H100 HBM peak, i.e. the bandwidth roofline for this traffic.")


def extras(T, L, timer, dtype=torch.bfloat16):
    D = K3["hidden"]
    set_backend("triton")
    q, w, ow, res = make_inputs(T, D, L, dtype)
    do = torch.randn_like(res[0])

    print("=" * 96)
    print(f"Extras @ K3 shape T={T}, D={D}, L={L}, bf16")
    print("=" * 96)

    # 1) output-RMSNorm fusion: fused(kernel) vs fused(no onorm) + separate rms_norm
    fused_only = lambda: call_fused(q, w, ow, res, True, 1)  # noqa: E731
    # unfused path: bf16 mix output, then a standalone RMSNorm (what a non-fused
    # framework does); torch's rms_norm upcasts to fp32 internally on bf16 input.
    split = lambda: torch.nn.functional.rms_norm(  # noqa: E731
        call_fused(q, w, ow, res, False, 1), (D,), ow, 1e-6)
    # ideal unfused variant: bf16 norm weight lets torch take its fused rms_norm path
    ow16 = ow.to(dtype)
    split16 = lambda: torch.nn.functional.rms_norm(  # noqa: E731
        call_fused(q, w, ow, res, False, 1), (D,), ow16, 1e-6)
    d_fused, _ = timer.device_ms(fused_only)
    d_split, k_split = timer.device_ms(split)
    d_split16, _ = timer.device_ms(split16)
    gb = 2 * T * D * 2 / 1e9
    print(f"[norm fusion] fused(onorm)      : {d_fused * 1e3:7.0f} us")
    print(f"[norm fusion] fused + rms_norm(bf16 w): {d_split16 * 1e3:7.0f} us "
          f"(+{(d_split16 - d_fused) * 1e3:.0f} us = the pass that fusion removes, "
          f"{gb / max(d_split16 - d_fused, 1e-6) * 1e3:.0f} GB/s)")
    delta = max(d_split - d_fused, 1e-6)
    print(f"[norm fusion] fused + rms_norm  : {d_split * 1e3:7.0f} us "
          f"(+{delta * 1e3:.0f} us = {gb / delta * 1e3:.0f} GB/s for "
          f"an extra {gb:.2f} GB pass) | {top_kernels(k_split)}")
    print()

    # 2) checkpoint_level: 0 keeps o_pre (faster bwd, more memory) vs 1 recomputes
    print(f"{'ckpt':>5} {'bwd_dev_us':>11} {'bwd_GB/s':>9} {'peak_MB':>8}")
    for ck in (0, 1):
        qq, ww, oww, rr = make_inputs(T, D, L, dtype)
        oo = call_fused(qq, ww, oww, rr, True, ck)
        bb = lambda: torch.autograd.grad(oo, [*rr, qq, ww, oww], do, retain_graph=True)  # noqa: E731
        bb()
        dev, _ = timer.device_ms(bb)
        peak = peak_mem_mb(lambda: torch.autograd.grad(
            call_fused(qq, ww, oww, rr, True, ck), [*rr, qq, ww, oww], do))
        print(f"{ck:>5} {dev * 1e3:>11.0f} "
              f"{bytes_bwd(T, D, L, 2, True, ck) / (dev / 1e3) / 1e9:>9.0f} {peak:>8.0f}")
    print("ckpt=0 saves the pre-norm mixed residual; ckpt=1 (Megatron default) recomputes it,")
    print("trading one extra L*(N*D) read for N*D of activation memory.")
    print()

    # 3) compile bucket: L=8 -> L2=8, L=9 -> L2=16 (FLA pads the pointer table to >=8/16)
    print(f"{'L':>3} {'L2':>3} {'fwd_us':>8} {'bwd_us':>8} {'note':>34}")
    for LL in (8, 9):
        qq, ww, oww, rr = make_inputs(T, D, LL, dtype)
        doo = torch.randn_like(rr[0])
        oo = call_fused(qq, ww, oww, rr, True, 1)
        f, _ = timer.device_ms(lambda: call_fused(qq, ww, oww, rr, True, 1))  # noqa: E731
        bb = lambda: torch.autograd.grad(oo, [*rr, qq, ww, oww], doo, retain_graph=True)  # noqa: E731
        b, _ = timer.device_ms(bb)
        note = "same L2 bucket" if LL == 8 else "crosses to L2=16 bucket (recompile)"
        print(f"{LL:>3} {max(8, 1 << (LL - 1).bit_length()):>3} {f * 1e3:>8.0f} {b * 1e3:>8.0f} {note:>34}")
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["check", "sweep", "schedule", "extras", "all"],
                    default="all")
    ap.add_argument("--fla-src", default=None, help="FLA checkout to import (or $FLA_SRC)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--seqs", default="1024,4096,8192,16384")
    ap.add_argument("--Ls", default="2,3,5,8,9")
    ap.add_argument("--dtypes", default="bfloat16")
    ap.add_argument("--backends", default="triton,gluon,naive")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--no-onorm", action="store_true", help="sweep without the fused output RMSNorm")
    ap.add_argument("--ckpt", type=int, choices=[0, 1], default=1)
    ap.add_argument("--naive-max-tokens", type=int, default=8192)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available", file=sys.stderr)
        return 1

    ensure_fla(args.fla_src)
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
          f"cap {torch.cuda.get_device_capability(0)} | "
          f"FLA_ATTNRES_GLUON={os.environ.get('FLA_ATTNRES_GLUON')}")

    seqs = [int(s) for s in args.seqs.split(",") if s]
    Ls = [int(s) for s in args.Ls.split(",") if s]
    backends = [b for b in args.backends.split(",") if b]
    dtypes = [getattr(torch, d) for d in args.dtypes.split(",") if d]
    onorm = not args.no_onorm
    rows = []
    timer = KernelTimer(warmup=args.warmup, repeat=args.repeat)

    if args.mode in ("check", "all"):
        check(seqs=seqs[:1], Ls=(Ls[0], Ls[-1]),
              backends=[b for b in backends if b != "naive"])
        print()

    if args.mode in ("sweep", "all"):
        for dt in dtypes:
            sweep(seqs, Ls, backends, timer, rows, dt, onorm, args.ckpt,
                  args.naive_max_tokens)
            print()

    if args.mode in ("schedule", "all"):
        schedule_mode(seqs, timer, dtypes[0], onorm, args.ckpt)
        print()

    if args.mode in ("extras", "all"):
        extras(max(seqs), max(Ls), timer, dtypes[0])

    if args.csv and rows:
        cols = ["backend", "dtype", "T", "D", "L", "onorm", "ckpt", "fwd_dev", "fwd_wall",
                "fwd_gbps", "bwd_dev", "bwd_wall", "bwd_gbps", "step_dev", "step_wall",
                "peak_mb", "fwd_kernels", "bwd_kernels", "error"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                            for k, v in r.items()})
        print(f"\nCSV written: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
