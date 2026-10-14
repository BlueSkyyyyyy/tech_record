"""P4-2: FlashKDA forward vs the fp64 token-recurrent golden (``src/kda_ref.py``).

End-to-end numerical comparison of the compiled FlashKDA kernel (bf16, CHUNK=16,
D=128) against the pure-PyTorch fp64 gated-delta-rule recurrence, over multiple
sequence lengths (incl. non-multiples of 16 and multi-batch), head counts, seeds
and gate regimes.

Golden:
  * q,k L2-normalised in fp64 (kernel does a warp-tree norm in fp32 on bf16 input),
  * gate activation ``g = lower_bound*log2e * sigmoid(exp(A_log)*(g_logits+dt_bias))``,
  * per-token recurrence ``S_t = diag(2^g_t) S_{t-1} + k_t ⊗ β_t (v_t - (k_t*2^g_t) S_{t-1})``.

FlashKDA state is stored as ``Sᵀ`` ([V,K], matching the K2 ``s_acc`` layout and the
wrapper docstring ``[N,H,V,K]``); the golden uses ``S`` ([K,V]). We therefore pass
``initial_state = h0ᵀ`` and compare ``final_state`` against ``Sᵀ``.

Usage (inside kernel_lab):
    python ~/proj/tech_record/code/flash-kda/harness/check_fwd_numerics.py
"""

import os
import math
import sys

import torch

SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, SRC)
from kda_ref import kda_gate, kda_token_recurrent, l2_normalize  # noqa: E402

LOG2E = 1.4426950408889634


def stats(label, gold, pred):
    d = (gold.float() - pred.float()).abs()
    ref = gold.float().abs()
    max_abs = d.max().item()
    max_rel = (d.max() / (ref.max() + 1e-12)).item()
    mean_abs = d.mean().item()
    print(f"  {label:26s} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  mean_abs={mean_abs:.3e}")
    return dict(max_abs=max_abs, max_rel=max_rel)


def run_case(name, B, T, H, g_scale, seed, lower_bound=-5.0):
    import flash_kda

    D = 128
    dev = "cuda"
    scale = 1.0 / math.sqrt(D)
    torch.manual_seed(seed)

    q = torch.randn(B, T, H, D, device=dev)
    k = torch.randn(B, T, H, D, device=dev)
    v = torch.randn(B, T, H, D, device=dev)
    beta = torch.randn(B, T, H, device=dev)
    g = torch.randn(B, T, H, D, device=dev) * g_scale
    A_log = torch.randn(H, device=dev)
    dt_bias = torch.randn(H, D, device=dev)
    h0 = torch.randn(B, H, D, D, device=dev)

    # ---- fp64 golden ----
    o_gold = torch.empty(B, T, H, D, device=dev, dtype=torch.float64)
    s_gold = torch.empty(B, H, D, D, device=dev, dtype=torch.float64)
    g_act = kda_gate(g.double(), A_log.double(), dt_bias.double(), lower_bound)  # [B,T,H,D]
    for b in range(B):
        for h in range(H):
            qn = l2_normalize(q[b, :, h].double())
            kn = l2_normalize(k[b, :, h].double())
            o, s = kda_token_recurrent(
                qn, kn, v[b, :, h].double(), g_act[b, :, h], torch.sigmoid(beta[b, :, h].double()),
                scale, h0[b, h].double())
            o_gold[b, :, h] = o.float()
            s_gold[b, h] = s.float()

    # ---- FlashKDA kernel ----
    out = torch.empty(B, T, H, D, device=dev, dtype=torch.bfloat16)
    fin = torch.empty(B, H, D, D, device=dev, dtype=torch.bfloat16)
    flash_kda.fwd(
        q.bfloat16().contiguous(), k.bfloat16().contiguous(), v.bfloat16().contiguous(),
        g.bfloat16().contiguous(), beta.bfloat16().contiguous(), scale, out,
        A_log=A_log.clone(), dt_bias=dt_bias.clone(), lower_bound=lower_bound,
        initial_state=h0.bfloat16().transpose(-1, -2).contiguous(), final_state=fin,
    )
    torch.cuda.synchronize()

    print(f"\n{'='*74}\nCase [{name}]  B={B} T={T} H={H} g_scale={g_scale} seed={seed}")
    r_o = stats("o    kernel vs gold", o_gold, out)
    r_sT = stats("S^T  kernel vs gold", s_gold.transpose(-1, -2), fin)  # kernel stores [V,K]
    r_s = stats("S (unchained) check", s_gold, fin.transpose(-1, -2))
    return dict(o=r_o, sT=r_sT, s=r_s)


def main():
    cases = []
    # shape sweep (T non-multiple of 16 included), H sweep, multi-batch, seeds
    for T in [16, 17, 64, 128, 512, 1024]:
        cases.append((f"T={T}", 1, T, 4, 2.0, 0))
    for H in [1, 2, 8]:
        cases.append((f"H={H}", 1, 256, H, 2.0, 0))
    cases.append(("B=3", 3, 128, 2, 2.0, 0))
    for seed in [1, 2, 3]:
        cases.append((f"seed={seed}", 1, 256, 4, 2.0, seed))
    # gate regimes
    cases.append(("g=0 (no decay)", 1, 256, 4, 0.0, 0))
    cases.append(("g=N(0,1)", 1, 256, 4, 1.0, 0))
    cases.append(("g=N(0,4)", 1, 256, 4, 4.0, 0))
    cases.append(("T=1023", 1, 1023, 2, 2.0, 7))

    agg = []
    for name, B, T, H, gs, seed in cases:
        agg.append((name, run_case(name, B, T, H, gs, seed)))

    print(f"\n{'='*74}\nSUMMARY  (o and S as S^T, the kernel layout)")
    print(f"  {'case':16s} {'o.max_abs':>12s} {'o.max_rel':>12s} "
          f"{'S.max_abs':>12s} {'S.max_rel':>12s}")
    for name, r in agg:
        print(f"  {name:16s} {r['o']['max_abs']:12.3e} {r['o']['max_rel']:12.3e} "
              f"{r['sT']['max_abs']:12.3e} {r['sT']['max_rel']:12.3e}")
    print("\nDONE")


if __name__ == "__main__":
    main()
