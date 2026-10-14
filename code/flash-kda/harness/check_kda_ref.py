"""P1-4: cross-check ``src/kda_ref.py`` against FLA ``chunk_kda`` on GPU.

Runs the CPU pure-PyTorch reference (token-recurrent golden + chunked) side by side
with FLA's Triton ``chunk_kda`` on the *same* inputs (raw q/k logits; gate via
``use_gate_in_kernel``), reporting max abs / rel for output ``o`` and final state.

Usage (inside kernel_lab):
    python /ssd/.../code/flash-kda/harness/check_kda_ref.py
"""

import os
import math
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/proj/tech_record/code/flash-kda/src"))
from kda_ref import kda_chunked_fwd, kda_token_recurrent, l2_normalize  # noqa: E402


def stats(label, gold, pred):
    d = (gold.float() - pred.float()).abs()
    ref = gold.float().abs()
    print(f"  {label:26s} max_abs={d.max().item():.3e}  "
          f"max_rel={(d.max() / (ref.max() + 1e-12)).item():.3e}  "
          f"mean_abs={d.mean().item():.3e}")


def run_case(name, q, k, v, g, beta, h0, A_log, dt_bias, scale, lower_bound):
    from fla.ops.kda import chunk_kda

    B, T, H, D = q.shape
    print(f"\n{'='*70}\nCase [{name}]  B={B} T={T} H={H} D={D}")

    o_fla, s_fla = chunk_kda(
        q=q.clone().bfloat16(), k=k.clone().bfloat16(), v=v.clone().bfloat16(),
        g=g.clone().bfloat16(), beta=beta.clone().bfloat16(), scale=scale,
        initial_state=h0.clone().float(), output_final_state=True,
        use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        A_log=A_log.clone(), dt_bias=dt_bias.clone(), lower_bound=lower_bound,
        # default layout: final_state is [N, H, K, V] == our S (K-major).
        # (transpose_state_layout=True would instead reinterpret initial_state as V-first.)
    )
    o_fla = o_fla.float()
    s_fla = s_fla.float()

    print(f"  scale(o): ref|o|~{q.shape[0]}  fla|o|max={o_fla.abs().max().item():.3f}  "
          f"fla|s|max={s_fla.abs().max().item():.3f}")

    for h in range(H):
        qn = l2_normalize(q[0, :, h].reshape(-1, D)).reshape(T, D)
        kn = l2_normalize(k[0, :, h].reshape(-1, D)).reshape(T, D)
        # gate: build log2-decay for the reference (matches FLA use_gate_in_kernel).
        g_act = (lower_bound * 1.4426950408889634) * torch.sigmoid(
            torch.exp(A_log[h]) * (g[0, :, h].float() + dt_bias[h].view(D)))
        beta_act = torch.sigmoid(beta[0, :, h].float())

        o_rec, s_rec = kda_token_recurrent(
            qn.double(), kn.double(), v[0, :, h].double(), g_act.double(),
            beta_act.double(), scale, h0[0, h].double())
        o_chk, s_chk = kda_chunked_fwd(
            qn.double(), kn.double(), v[0, :, h].double(), g_act.double(),
            beta_act.double(), scale, h0[0, h].double(), chunk=16)
        o_rec, o_chk = o_rec.float(), o_chk.float()

        of = o_fla[0, :, h]
        sf = s_fla[0, h]
        print(f"  -- head {h} --")
        stats("o token vs fla", o_rec, of)
        stats("o chunk  vs fla", o_chk, of)
        stats("S token vs fla", s_rec, sf)
        stats("S chunk vs fla", s_chk, sf)


def main():
    torch.manual_seed(0)
    lower_bound = -5.0
    D = 128
    H = 2
    B = 1
    scale = 1.0 / math.sqrt(D)
    dev = "cuda"
    for T in [20, 64]:
        q = torch.randn(B, T, H, D, device=dev)
        k = torch.randn(B, T, H, D, device=dev)
        v = torch.randn(B, T, H, D, device=dev)
        beta = torch.randn(B, T, H, device=dev)
        g = torch.randn(B, T, H, D, device=dev) * 2
        h0 = torch.randn(B, H, D, D, device=dev)
        A_log = torch.randn(H, device=dev)
        dt_bias = torch.randn(H, D, device=dev)
        run_case(f"T={T} g=N(0,2)", q, k, v, g, beta, h0, A_log, dt_bias, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
