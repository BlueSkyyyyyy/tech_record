# Copyright (c) 2026. All rights reserved.
#
# KDA numerical-precision comparison: FlashKDA vs FLA Triton (vs an fp64 golden).
#
# For each Kimi K3 KDA shape (B=1, H=96, K=V=128; safe gate, lower_bound=-5),
# we feed the *same* bf16 inputs to:
#   - FLA `chunk_kda` Triton backend
#   - FlashKDA backend (via FLA dispatch)
# and compare each output `o` against a float64 token-recurrence golden computed on
# the same (bf16-quantized) inputs. This isolates the kernels' ARITHMETIC error from
# the (common) bf16 input-quantization error.
#
# Metrics per case:
#   max_abs  = max |o_impl - o_gold|
#   max_rel  = max |o_impl - o_gold| / (|o_gold| + eps)   (per-element)
#   rms      = sqrt(mean((o_impl - o_gold)^2))
#   ref_rel  = max_abs / max|o_gold|                       (signal-normalized)
# and the mutual difference FlashKDA vs Triton.
#
# Usage (inside kernel_lab, on an idle GPU):
#   python check_kda_precision.py
#   python check_kda_precision.py --seqs 1024,4096,8192 --seeds 0,1,2
#
# NOTE: local absolute paths are intentionally avoided in this file.

from __future__ import annotations

import argparse
import sys

import torch

import bench_kda_kimi_k3 as B  # reuse inputs / call / backend switch


# --------------------------------------------------------------------------- #
# fp64 golden: exact KDA math on the bf16-quantized inputs
# --------------------------------------------------------------------------- #
def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.pow(2).sum(-1, keepdim=True) + eps).sqrt()


def golden_o(q, k, v, g_raw, beta_raw, A_log, dt_bias, scale, lower_bound=-5.0):
    """Token-recurrence KDA in float64. Inputs are the bf16 tensors the kernels get."""
    q, k, v, g_raw, beta_raw = (x.double() for x in (q, k, v, g_raw, beta_raw))
    A_log, dt_bias = A_log.double(), dt_bias.double()
    B_, T, H, K = q.shape
    V = v.shape[-1]
    # safe gate: g = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))
    z = g_raw + dt_bias.view(H, -1)
    gate = lower_bound * torch.sigmoid(torch.exp(A_log).view(H, 1) * z)   # [B,T,H,K]
    beta = torch.sigmoid(beta_raw)                                        # [B,T,H]
    qn, kn = _l2norm(q), _l2norm(k)
    S = torch.zeros(B_, H, K, V, dtype=torch.float64, device=q.device)
    o = torch.empty(B_, T, H, V, dtype=torch.float64, device=q.device)
    for t in range(T):
        a = torch.exp(gate[:, t])                       # [B,H,K]
        p = torch.einsum("bhk,bhkv->bhv", a * kn[:, t], S)
        d = beta[:, t].unsqueeze(-1) * (v[:, t] - p)
        S = a.unsqueeze(-1) * S + kn[:, t].unsqueeze(-1) * d.unsqueeze(-2)
        o[:, t] = scale * torch.einsum("bhk,bhkv->bhv", qn[:, t], S)
    return o, S


def _metrics(x, gold):
    x, gold = x.double(), gold
    diff = (x - gold).abs()
    max_abs = diff.max().item()
    gmax = gold.abs().max().item()
    ref_rel = max_abs / (gmax + 1e-12)
    rms = (x - gold).pow(2).mean().sqrt().item()
    # per-element relative error, restricted to elements where the golden signal is
    # non-negligible (>1% of the peak), otherwise dividing by ~0 explodes.
    mask = gold.abs() > 1e-2 * gmax
    rel_filt = (diff[mask] / gold.abs()[mask]).max().item() if mask.any() else float("nan")
    return max_abs, ref_rel, rms, rel_filt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="512,1024,4096,8192,16384")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--D", type=int, default=128)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available", file=sys.stderr)
        return 1

    seqs = [int(s) for s in args.seqs.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    H, D = args.H, args.D
    scale = D ** -0.5

    print("=" * 110)
    print("KDA precision: FlashKDA vs FLA Triton vs fp64 golden")
    print(f"shape: B=1 H={H} K=V={D}  bf16 inputs, safe gate (lower_bound=-5), in-kernel gate/l2norm/beta")
    print("=" * 110)
    sub = (f"{'T':>7} {'seed':>4} | {'Triton bf16':^34} | {'FlashKDA bf16':^34} | {'Triton fp32':^22} | "
           f"{'FK vs Tri':>9}")
    print(sub)
    sub2 = (f"{'':>12} | {'max_abs':>9}{'refrel':>9}{'rms':>8}{'relf':>8} | "
            f"{'max_abs':>9}{'refrel':>9}{'rms':>8}{'relf':>8} | "
            f"{'max_abs':>9}{'refrel':>9}{'':>4} | {'max_abs':>9}")
    print(sub2)
    print("-" * len(sub2))

    for T in seqs:
        for seed in seeds:
            # fp32 source -> bf16 kernel inputs
            q32, k32, v32, g32, beta32, A_log, dt_bias = B.make_inputs(1, T, H, D, D, torch.float32, seed)
            qb, kb, vb, gb, bb = [x.to(torch.bfloat16) for x in (q32, k32, v32, g32, beta32)]

            gold, _ = golden_o(qb, kb, vb, gb, bb, A_log, dt_bias, scale)

            # Triton bf16
            B.use_backend("triton")
            with torch.no_grad():
                o_tri, _ = B.call_kda(qb, kb, vb, gb, bb, A_log, dt_bias)
            # FlashKDA bf16
            B.use_backend("flashkda")
            with torch.no_grad():
                o_fk, _ = B.call_kda(qb, kb, vb, gb, bb, A_log, dt_bias)
            # Triton fp32 (kernel arithmetic floor reference)
            use32 = True
            try:
                B.use_backend("triton")
                with torch.no_grad():
                    o_tri32, _ = B.call_kda(q32, k32, v32, g32, beta32, A_log, dt_bias)
                gold32, _ = golden_o(q32, k32, v32, g32, beta32, A_log, dt_bias, scale)
            except Exception:  # noqa: BLE001
                use32 = False

            mt = _metrics(o_tri, gold)
            mf = _metrics(o_fk, gold)
            muv = (o_tri.double() - o_fk.double()).abs().max().item()
            if use32:
                m32 = _metrics(o_tri32, gold32)
                s32 = f"{m32[0]:>9.2e}{m32[1]:>9.1e}{'':>4}"
            else:
                s32 = f"{'-':>22}"
            print(f"{T:>7} {seed:>4} | {mt[0]:>9.2e}{mt[1]:>9.1e}{mt[2]:>8.1e}{mt[3]:>8.1e} | "
                  f"{mf[0]:>9.2e}{mf[1]:>9.1e}{mf[2]:>8.1e}{mf[3]:>8.1e} | {s32:>22} | {muv:>9.2e}")

    print("-" * len(sub2))
    print("refrel = max_abs / max|golden|;  relf = max per-element rel over |golden| > 1% peak;")
    print("rms = element-wise RMS error.  golden = fp64 token recurrence on the SAME bf16 inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
