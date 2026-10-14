# Copyright (c) 2026. All rights reserved.
#
# Benchmark KDA (Kimi Delta Attention) on the *actual Kimi K3* shapes, comparing:
#   - FLA `chunk_kda` (Triton)            -> the default/fallback path (fwd + bwd)
#   - FLA `chunk_kda` (TileLang backend)  -> another FLA backend (fwd + bwd)
#   - FlashKDA (Moonshot CUTLASS backend) -> inference-only, auto-dispatched by FLA
#
# Kimi K3 model shapes (official text_config):
#   93 layers = 69 KDA + 24 full-MLA.
#   KDA layer: num_heads = 96, head_dim (K = V) = 128, gate_lower_bound = -5.0.
#   => every KDA tensor is [B, T, H=96, D=128]  (g: [B,T,96,128], beta: [B,T,96]).
#
# What this script reports
#   1) `--mode sweep`   : forward / backward / fwd+bwd, each measured SEPARATELY
#                         (backward-only is timed via autograd.grad(retain_graph=True)
#                          on a pre-built graph, so it excludes the forward).
#   2) `--mode support` : a dtype x backend x {fwd,bwd} support matrix. It reports the
#                         **effective** backend (what FLA's dispatch actually picks),
#                         because a rejected backend silently falls back to Triton.
#   3) `--mode check`   : correctness Triton vs FlashKDA (bf16).
#
# Timing: device_ms = PURE device kernel time (CUPTI / torch.profiler), launch excluded.
#         wall_ms   = CUDA-event mean over `repeat` calls, launch included.
#
# Backend selection via FLA runtime env (read per call by fla/ops/backends):
#   FLA_TILELANG / FLA_FLASH_KDA -> "0" disables; if none enabled, dispatch falls back
#   to the default Triton `chunk_kda`.
#
# Usage (inside the kernel_lab container, on an idle GPU):
#   python bench_kda_kimi_k3.py --mode support
#   python bench_kda_kimi_k3.py --mode sweep --dtypes bfloat16 --seqs 1024,4096,8192,32768
#   python bench_kda_kimi_k3.py --mode check
#
# NOTE: local absolute paths are intentionally avoided in this file.

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

H100 = {"bf16_tflops": 989.4, "bandwidth": 3.35e12}


# --------------------------------------------------------------------------- #
# Timer
# --------------------------------------------------------------------------- #
class KernelTimer:
    def __init__(self, warmup: int = 5, repeat: int = 20):
        self.warmup = warmup
        self.repeat = repeat
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def wall_ms(self, fn):
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

    def device_ms(self, fn):
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(self.repeat):
                fn()
            torch.cuda.synchronize()
        total_us = sum(evt.device_time_total for evt in prof.key_averages())
        return total_us / self.repeat / 1e3


# --------------------------------------------------------------------------- #
# Backend selection + effective-backend detection
# --------------------------------------------------------------------------- #
def use_backend(name: str) -> None:
    """name in {triton, tilelang, flashkda}."""
    os.environ["FLA_TILELANG"] = "1" if name == "tilelang" else "0"
    os.environ["FLA_FLASH_KDA"] = "1" if name == "flashkda" else "0"


def effective_backend(verify_kwargs: dict) -> str:
    """Which FLA backend dispatch actually picks for these call args (post-verify)."""
    from fla.ops.kda.backends import kda_registry

    for b in kda_registry._get_sorted_backends():
        if not (b.is_available() and b.is_enabled()):
            continue
        try:
            ok, _ = b.verify("chunk_kda", **verify_kwargs)
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            return b.backend_type
    return "triton"


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def make_inputs(B, T, H, K, V, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    # fp8 dtypes are not accepted by torch.randn -> draw in fp32 then cast.
    def f(*s):
        return torch.randn(*s, device=dev, dtype=torch.float32).to(dtype)
    q = f(B, T, H, K)
    k = f(B, T, H, K)
    v = f(B, T, H, V)
    g = f(B, T, H, K)
    beta = f(B, T, H)
    A_log = torch.randn(H, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(H * K, device=dev, dtype=torch.float32)
    return q, k, v, g, beta, A_log, dt_bias


# Kimi K3 KDA: safe gate, lower_bound=-5, full in-kernel gate + qk-l2norm + beta-sigmoid.
KDA_KW = dict(
    use_gate_in_kernel=True,
    use_qk_l2norm_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    safe_gate=True,
    lower_bound=-5.0,
    state_v_first=True,
)
# extra kwargs only used for the verifier (its signature lists them explicitly)
_VERIFY_EXTRA = dict(
    scale=None, initial_state=None, output_final_state=False, allow_neg_eigval=False,
    disable_recompute=False, return_intermediate_states=False, cu_seqlens=None,
    cu_seqlens_cpu=None, cp_context=None, use_graph=False, max_num_seqs=None,
)


def call_kda(q, k, v, g, beta, A_log, dt_bias, scale=None):
    from fla.ops.kda import chunk_kda

    return chunk_kda(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias,
                     scale=scale, initial_state=None, output_final_state=False, **KDA_KW)


def _verify_kwargs(q, k, v, g, beta, A_log, dt_bias):
    return dict(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias, **_VERIFY_EXTRA, **KDA_KW)


# --------------------------------------------------------------------------- #
# FLOPs / bytes (analytic, for TF/s and GB/s columns)
# --------------------------------------------------------------------------- #
def kda_flops_bytes(B, T, H, K, V, chunk=64):
    import math
    C = chunk
    nchunk = math.ceil(T / C)
    per_chunk = 4 * K * C * C + 6 * K * C * V
    flops = 2 * (B * H * nchunk) * per_chunk
    elems = B * T * H * (3 * K + 2 * V + 1)
    return flops, elems * 2


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #
def measure_fwd(q, k, v, g, beta, A_log, dt_bias, timer):
    with torch.no_grad():
        dev = timer.device_ms(lambda: call_kda(q, k, v, g, beta, A_log, dt_bias))
        wall = timer.wall_ms(lambda: call_kda(q, k, v, g, beta, A_log, dt_bias))
    return dev, wall


def measure_bwd(q, k, v, g, beta, A_log, dt_bias, timer):
    """Backward-ONLY device time.

    The forward (with a grad graph) is built ONCE outside the timed region; the timed
    loop then executes only ``autograd.grad(..., retain_graph=True)``, i.e. only the
    backward kernels (including the forward-recompute that FLA does inside backward).
    This isolates the backward better than ``(fwd+bwd) - fwd``, which overestimates
    because the grad-enabled forward is more expensive than the ``no_grad`` forward.
    (Run on an idle GPU; under contention this can be inflated.)
    """
    inp = [x.clone().requires_grad_(True) for x in (q, k, v, g, beta)]
    o, _ = call_kda(*inp, A_log, dt_bias)
    do = torch.randn_like(o)
    return timer.device_ms(lambda: torch.autograd.grad(o, inp, do, retain_graph=True))


def measure_fwd_bwd(q, k, v, g, beta, A_log, dt_bias, timer):
    """Full training step (fresh graph + forward + backward), CUPTI device time."""
    def step():
        inp = [x.clone().requires_grad_(True) for x in (q, k, v, g, beta)]
        o, _ = call_kda(*inp, A_log, dt_bias)
        torch.autograd.grad(o.float().sum(), inp)
    return timer.device_ms(step)


# --------------------------------------------------------------------------- #
# Support matrix
# --------------------------------------------------------------------------- #
def support_matrix(dtypes, T=2048):
    print("=" * 96)
    print("KDA backend x dtype support matrix (effective backend = what FLA dispatch picks)")
    print(f"shape: B=1 T={T} H=96 K=V=128  flags: safe_gate + in-kernel gate/l2norm/beta/beta")
    print("=" * 96)
    print(f"{'dtype':>10} | {'requested':>10} | {'fwd effective':>14} | {'bwd effective':>14}")
    print("-" * 62)
    rows = []
    for dt_name in dtypes:
        dt = getattr(torch, dt_name)
        for name in ("triton", "tilelang", "flashkda"):
            use_backend(name)
            q, k, v, g, beta, A_log, dt_bias = make_inputs(1, T, 96, 128, 128, dt)
            vkw = _verify_kwargs(q, k, v, g, beta, A_log, dt_bias)
            with torch.no_grad():
                eff_fwd = effective_backend(vkw)
            with torch.enable_grad():
                eff_bwd = effective_backend(vkw)
            # runtime smoke
            rt_fwd = _smoke_fwd(q, k, v, g, beta, A_log, dt_bias)
            rt_bwd = _smoke_bwd(q, k, v, g, beta, A_log, dt_bias)
            print(f"{dt_name:>10} | {name:>10} | {eff_fwd+'/'+rt_fwd:>14} | {eff_bwd+'/'+rt_bwd:>14}")
            rows.append((dt_name, name, eff_fwd, rt_fwd, eff_bwd, rt_bwd))
    print("-" * 62)
    print("cell = <effective backend>/<runtime status>; runtime: ok | FAIL:<msg>")
    print("rule: a requested backend that rejects is silently replaced by Triton;")
    print("      FlashKDA = bf16 + inference only; TileLang fp32 bwd fails to compile.")
    return rows


def _smoke_fwd(q, k, v, g, beta, A_log, dt_bias):
    try:
        with torch.no_grad():
            call_kda(q, k, v, g, beta, A_log, dt_bias)
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"FAIL:{type(e).__name__}:{str(e)[:28]}"


def _smoke_bwd(q, k, v, g, beta, A_log, dt_bias):
    try:
        inp = [x.clone().requires_grad_(True) for x in (q, k, v, g, beta)]
        o, _ = call_kda(*inp, A_log, dt_bias)
        torch.autograd.grad(o.float().sum(), inp)
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"FAIL:{type(e).__name__}:{str(e)[:28]}"


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #
def check(seqs=(2048,)):
    print("=" * 78)
    print("KDA correctness: FLA Triton vs FlashKDA (bf16)")
    print("=" * 78)
    ok = True
    for T in seqs:
        q, k, v, g, beta, A_log, dt_bias = make_inputs(1, T, 96, 128, 128)
        use_backend("triton")
        with torch.no_grad():
            o_tri, _ = call_kda(q, k, v, g, beta, A_log, dt_bias)
        use_backend("flashkda")
        try:
            with torch.no_grad():
                o_fk, _ = call_kda(q, k, v, g, beta, A_log, dt_bias)
        except Exception as e:  # noqa: BLE001
            print(f"  T={T}: FlashKDA FAILED: {e}")
            ok = False
            continue
        num = (o_tri.float() - o_fk.float()).abs().max().item()
        den = o_tri.float().abs().max().item() + 1e-9
        print(f"  T={T}: max|o_tri - o_fk| = {num:.3e}   rel = {num/den:.3e}")
    return ok


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def sweep(dtypes, seqs, backends, timer, rows_out):
    B, H, K, V = 1, 96, 128, 128
    print("=" * 120)
    print(f"Kimi K3 KDA: B={B} H={H} K={K} V={V}   (fwd / bwd / fwd+bwd measured separately, CUPTI device)")
    print("=" * 120)
    hdr = (f"{'dtype':>9} {'req':>8} {'eff':>8} {'T':>7} | "
           f"{'fwd dev':>9} {'fwd wall':>9} {'TF/s':>6} | {'bwd dev':>9} | {'f+b dev':>9} | note")
    print(hdr)
    print("-" * len(hdr))
    for dt_name in dtypes:
        dt = getattr(torch, dt_name)
        for T in seqs:
            q, k, v, g, beta, A_log, dt_bias = make_inputs(B, T, H, K, V, dt)
            flops, nbytes = kda_flops_bytes(B, T, H, K, V)
            for name in backends:
                use_backend(name)
                vkw = _verify_kwargs(q, k, v, g, beta, A_log, dt_bias)
                with torch.no_grad():
                    eff = effective_backend(vkw)
                note = "" if eff == name else f"->{eff}"
                try:
                    fwd_dev, fwd_wall = measure_fwd(q, k, v, g, beta, A_log, dt_bias, timer)
                except Exception as e:  # noqa: BLE001
                    print(f"{dt_name:>9} {name:>8} {eff:>8} {T:>7} | fwd FAILED: {str(e)[:40]}")
                    continue
                # backward is measured in ISOLATION (forward built once outside the timed
                # loop, only autograd.grad timed). f+b is also measured for cross-reference.
                bwd_dev = f_b_dev = float("nan")
                if name == "flashkda":
                    # FlashKDA is inference-only; any grad-enabled call falls back to Triton,
                    # so a "FlashKDA backward" number would be meaningless.
                    note = (note + " bwd n/a (inference-only)").strip()
                elif T <= 32768:
                    try:
                        bwd_dev = measure_bwd(q, k, v, g, beta, A_log, dt_bias, timer)
                        f_b_dev = measure_fwd_bwd(q, k, v, g, beta, A_log, dt_bias, timer)
                    except Exception as e:  # noqa: BLE001
                        note = (note + f" bwdFAIL:{str(e)[:20]}").strip()
                print(f"{dt_name:>9} {name:>8} {eff:>8} {T:>7} | "
                      f"{fwd_dev:>9.3f} {fwd_wall:>9.3f} {flops/(fwd_dev/1e3)/1e12:>6.1f} | "
                      f"{bwd_dev:>9.3f} | {f_b_dev:>9.3f} | {note}")
                rows_out.append(dict(dtype=dt_name, requested=name, effective=eff, T=T,
                                     fwd_dev=fwd_dev, fwd_wall=fwd_wall, bwd_dev=bwd_dev,
                                     fwb_dev=f_b_dev, tflops=flops/(fwd_dev/1e3)/1e12,
                                     gb_s=nbytes/(fwd_dev/1e3)/1e9, note=note))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "support", "check", "all"], default="all")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--seqs", default="1024,4096,8192,16384,32768,65536")
    ap.add_argument("--dtypes", default="bfloat16")
    ap.add_argument("--backends", default="triton,flashkda,tilelang")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available", file=sys.stderr)
        return 1

    import fla  # noqa: F401
    print(f"fla {getattr(fla, '__version__', '?')} | torch {torch.__version__} | "
          f"{torch.cuda.get_device_name(0)}")

    if args.mode in ("check",):
        return 0 if check() else 1
    if args.mode in ("support",):
        support_matrix([d for d in args.dtypes.split(",") if d] or
                       ["bfloat16", "float16", "float32"])
        return 0

    dtypes = [d for d in args.dtypes.split(",") if d]
    seqs = [int(s) for s in args.seqs.split(",") if s]
    backends = [b for b in args.backends.split(",") if b]
    timer = KernelTimer(warmup=args.warmup, repeat=args.repeat)
    rows_out = []
    sweep(dtypes, seqs, backends, timer, rows_out)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dtype", "requested", "effective", "T", "H", "K", "V",
                        "fwd_dev_ms", "fwd_wall_ms", "bwd_dev_ms", "fwd_bwd_dev_ms",
                        "tflops", "gb_s", "note"])
            for r in rows_out:
                w.writerow([r["dtype"], r["requested"], r["effective"], r["T"], 96, 128, 128,
                            f"{r['fwd_dev']:.6f}", f"{r['fwd_wall']:.6f}",
                            f"{r['bwd_dev']:.6f}", f"{r['fwb_dev']:.6f}",
                            f"{r['tflops']:.3f}", f"{r['gb_s']:.1f}", r["note"]])
        print(f"\nCSV written: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
