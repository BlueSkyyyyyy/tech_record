"""P4-3: three-way forward comparison -- fp64 golden vs FlashKDA vs FLA chunk_kda.

On one set of inputs we run three implementations and report pairwise errors:

  * golden    : ``src/kda_ref.py::kda_token_recurrent`` in fp64 (per-token gated
                delta rule);
  * FlashKDA  : ``flash_kda.fwd`` (CUDA, SM90, bf16, CHUNK=16, D=128), state ``Sᵀ``;
  * FLA       : ``fla.ops.kda.chunk_kda`` (Triton, bf16, chunk_size=64), default
                state layout ``[N,H,K,V]`` == ``S``.

All three consume the *same* fp32 random tensors; the two bf16 implementations take
their bf16 rounding, the golden takes fp64. Reported metrics: max_abs and max_rel
(peak-relative, denominator = max|golden|), for output ``o`` and final state. States
are compared in the ``Sᵀ`` (``[V,K]``) layout for all pairs.

Usage (inside kernel_lab):
    python ~/proj/tech_record/code/flash-kda/harness/check_three_way.py
"""

import math
import sys

import torch

SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, SRC)
from kda_ref import kda_gate, kda_token_recurrent, l2_normalize  # noqa: E402


def pair(label, gold, pred):
    d = (gold.float() - pred.float()).abs()
    ref = gold.float().abs()
    max_abs = d.max().item()
    max_rel = (d.max() / (ref.max() + 1e-12)).item()
    mean_abs = d.mean().item()
    print(f"  {label:30s} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  mean_abs={mean_abs:.3e}")
    return dict(max_abs=max_abs, max_rel=max_rel, mean_abs=mean_abs)


def run_case(name, B, T, H, g_scale, seed, lower_bound=-5.0):
    import flash_kda
    from fla.ops.kda import chunk_kda

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

    # ---- fp64 golden (S: [K,V]) ----
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

    # ---- FlashKDA kernel (state S^T: [V,K]) ----
    out_fk = torch.empty(B, T, H, D, device=dev, dtype=torch.bfloat16)
    fin_fk = torch.empty(B, H, D, D, device=dev, dtype=torch.bfloat16)
    flash_kda.fwd(
        q.bfloat16().contiguous(), k.bfloat16().contiguous(), v.bfloat16().contiguous(),
        g.bfloat16().contiguous(), beta.bfloat16().contiguous(), scale, out_fk,
        A_log=A_log.clone(), dt_bias=dt_bias.clone(), lower_bound=lower_bound,
        initial_state=h0.bfloat16().transpose(-1, -2).contiguous(), final_state=fin_fk,
    )

    # ---- FLA chunk_kda (state [N,H,K,V] == S) ----
    o_fla, s_fla = chunk_kda(
        q=q.clone().bfloat16(), k=k.clone().bfloat16(), v=v.clone().bfloat16(),
        g=g.clone().bfloat16(), beta=beta.clone().bfloat16(), scale=scale,
        initial_state=h0.clone().float(), output_final_state=True,
        use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        A_log=A_log.clone(), dt_bias=dt_bias.clone(), lower_bound=lower_bound,
    )
    torch.cuda.synchronize()
    o_fla, s_fla = o_fla.float(), s_fla.float()

    print(f"\n{'='*78}\nCase [{name}]  B={B} T={T} H={H} g_scale={g_scale} seed={seed}")

    s_gold_T = s_gold.transpose(-1, -2)          # [V,K]
    s_fla_T = s_fla.transpose(-1, -2).float()    # [V,K]
    fin_fk_f = fin_fk.float()                    # already [V,K]

    print("  --- output o ---")
    r_of = pair("flash vs gold", o_gold, out_fk)
    r_oa = pair("fla   vs gold", o_gold, o_fla)
    r_ox = pair("flash vs fla ", o_fla, out_fk)
    print("  --- final state S^T ---")
    r_sf = pair("flash vs gold", s_gold_T, fin_fk_f)
    r_sa = pair("fla   vs gold", s_gold_T, s_fla_T)
    r_sx = pair("flash vs fla ", s_fla_T, fin_fk_f)
    return dict(o_flash=r_of, o_fla=r_oa, o_x=r_ox, s_flash=r_sf, s_fla=r_sa, s_x=r_sx)


def main():
    cases = []
    for T in [64, 128, 256, 512, 1024]:
        cases.append((f"T={T}", 1, T, 4, 2.0, 0))
    for H in [1, 2, 8]:
        cases.append((f"H={H}", 1, 256, H, 2.0, 0))
    cases.append(("B=3", 3, 128, 2, 2.0, 0))
    for seed in [1, 2, 3]:
        cases.append((f"seed={seed}", 1, 256, 4, 2.0, seed))
    cases.append(("g=0 (no decay)", 1, 256, 4, 0.0, 0))
    cases.append(("g=N(0,1)", 1, 256, 4, 1.0, 0))
    cases.append(("g=N(0,4)", 1, 256, 4, 4.0, 0))
    cases.append(("T=1023", 1, 1023, 2, 2.0, 7))

    agg = []
    for name, B, T, H, gs, seed in cases:
        agg.append((name, run_case(name, B, T, H, gs, seed)))

    print(f"\n{'='*78}\nSUMMARY  (max_abs / max_rel; states in S^T layout)")
    hdr = (f"  {'case':16s} | {'o fl-gold':>19s} | {'o fla-gold':>19s} | "
           f"{'o fl-fla':>19s} | {'S fl-gold':>19s} | {'S fla-gold':>19s} | {'S fl-fla':>19s}")
    print(hdr)
    for name, r in agg:
        def fmt(d):
            return f"{d['max_abs']:.2e}/{d['max_rel']:.2e}"
        print(f"  {name:16s} | {fmt(r['o_flash']):>19s} | {fmt(r['o_fla']):>19s} | "
              f"{fmt(r['o_x']):>19s} | {fmt(r['s_flash']):>19s} | {fmt(r['s_fla']):>19s} | "
              f"{fmt(r['s_x']):>19s}")
    print("\nDONE")


if __name__ == "__main__":
    main()
