"""Probe the layout convention of FlashKDA ``initial_state``/``final_state``.

A ``D x D`` recurrent state is square, so a K-major vs V-major mix-up never trips a
shape check -- it only shows up as a large error wherever the *initial state* still
influences the output. This script isolates it:

  * zero initial state  -> error is pure bf16 (~1e-4);
  * random initial state -> error is huge at t=0 but decays with the gate;
  * direct state compare -> ``S`` matches at ~0.5 but ``S^T`` matches at ~4e-3.

Conclusion (verified): FlashKDA stores ``S^T`` = ``[V, K]``; ``src/kda_ref.py`` uses
``S`` = ``[K, V]``. Evidence referenced by ``docs/04-numerics.md`` §3.

Usage (inside kernel_lab):
    python ~/proj/tech_record/code/flash-kda/harness/probe_state_layout.py
"""

import os
import math
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/proj/tech_record/code/flash-kda/src"))
from kda_ref import kda_gate, kda_token_recurrent, l2_normalize  # noqa: E402

import flash_kda  # noqa: E402


def main():
    D, T, H, B = 128, 64, 1, 1
    dev = "cuda"
    scale = 1 / math.sqrt(D)
    lb = -5.0
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, device=dev)
    k = torch.randn(B, T, H, D, device=dev)
    v = torch.randn(B, T, H, D, device=dev)
    beta = torch.randn(B, T, H, device=dev)
    g = torch.randn(B, T, H, D, device=dev) * 2
    A_log = torch.randn(H, device=dev)
    dt_bias = torch.randn(H, D, device=dev)

    ga = kda_gate(g.double(), A_log.double(), dt_bias.double(), lb)
    qn = l2_normalize(q[0, :, 0].double())
    kn = l2_normalize(k[0, :, 0].double())

    for h0mode in ["zero", "rand_correct", "rand_bug_untransposed"]:
        h0 = (torch.zeros(B, H, D, D, device=dev) if h0mode == "zero"
              else torch.randn(B, H, D, D, device=dev))
        o_g, s_g = kda_token_recurrent(
            qn, kn, v[0, :, 0].double(), ga[0, :, 0],
            torch.sigmoid(beta[0, :, 0].double()), scale, h0[0, 0].double())

        # correct call transposes the [K,V] state to the kernel's [V,K]; the "bug"
        # mode passes it untransposed to reproduce the layout mix-up.
        h0_kernel = h0.bfloat16()
        if h0mode != "rand_bug_untransposed":
            h0_kernel = h0_kernel.transpose(-1, -2)
        out = torch.empty(B, T, H, D, device=dev, dtype=torch.bfloat16)
        fin = torch.empty(B, H, D, D, device=dev, dtype=torch.bfloat16)
        flash_kda.fwd(
            q.bfloat16(), k.bfloat16(), v.bfloat16(), g.bfloat16(), beta.bfloat16(),
            scale, out, A_log=A_log, dt_bias=dt_bias, lower_bound=lb,
            initial_state=h0_kernel.contiguous(), final_state=fin)
        torch.cuda.synchronize()

        err = (o_g.float() - out[0, :, 0].float()).abs()
        per_t = err.max(dim=-1).values
        print(f"h0={h0mode}: o max_abs={err.max().item():.3e} mean={err.mean().item():.2e}")
        print("  per-token max err (first 8):", [f"{x:.2e}" for x in per_t[:8].tolist()])
        print("  S vs fin max_abs           :", (s_g.float() - fin[0, 0].float()).abs().max().item())
        print("  S^T vs fin max_abs         :", (s_g.float().t() - fin[0, 0].float()).abs().max().item())


if __name__ == "__main__":
    main()
