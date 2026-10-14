"""P4-6: FlashKDA's own *kernel-bit-exact* torch reference as a third oracle.

Motivation
----------
Every forward numeric anchor in this series so far used the **fp64 token-recurrent
golden** (``src/kda_ref.py``) or **FLA** ``chunk_kda``. Both are *independent* of the
FlashKDA kernel. FlashKDA itself ships ``tests/torch_ref.py`` — a pure-PyTorch
re-implementation that deliberately mirrors the kernel *bit-for-bit* (warp-tree L2
norm with FMA, ``tanh.approx`` sigmoid, FTZ ``exp2``, and the 8x8 forward-substitution
+ 16x16 bf16-HMMA inverse ``inv_fwd_subst_16``). FlashKDA's own ``tests/test_fwd.py:260``
asserts ``torch.equal(out_kernel, out_ref)`` — i.e. that reference is *exact*, not an
oracle at bf16 noise.

This harness uses it three ways:

A. **Oracle sanity / anchoring** — FlashKDA kernel vs its own ``torch_ref`` must be
   **bit-identical** on our shapes (reproduce their exact-match claim independently).
   This pins the reference as "the kernel's algebra", not "an approximation of it".
B. **Our derivation vs the kernel-faithful reference** — feed ``src/kda_ref.py``'s
   ``kda_chunked_fwd`` (P1-2's blocked algebra) the **same activated operands** the
   kernel-faithful reference consumes, and compare ``o`` / ``S^T``. Landing at the bf16
   floor (same band as kernel-vs-golden) validates the chunked formula independently of
   the token-recurrent golden.
C. **Exact-algebra equality** — run the same blocked loop in fp64 (no bf16 casts, exact
   sigmoid) against our ``kda_chunked_fwd`` in fp64: the two independent transcriptions
   of the chunked algebra must agree to ~machine precision.

Run inside `kernel_lab` (needs the built ``flash_kda_C`` + nvcc for the inline sigmoid):
    python ~/proj/tech_record/code/flash-kda/harness/check_fwd_torchref.py
"""

import math
import sys

import torch

SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, SRC)
from kda_ref import kda_chunked_fwd, l2_normalize  # noqa: E402

FLASHKDA = "~/github/FlashKDA"
sys.path.insert(0, FLASHKDA)
sys.path.insert(0, f"{FLASHKDA}/tests")
from torch_ref import torch_ref, l2_normalize_kernel_match, fp32_ex2_ftz, sigmoid_ext  # noqa: E402

LOG2E = 1.4426950408889634
D = 128
CHUNK = 16
LOWER_BOUND = -5.0


def stats(label, gold, pred):
    d = (gold.float() - pred.float()).abs()
    ref = gold.float().abs()
    max_abs = d.max().item()
    max_rel = (d.max() / (ref.max() + 1e-12)).item()
    print(f"  {label:40s} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")
    return dict(max_abs=max_abs, max_rel=max_rel)


def make_inputs(B, T, H, seed, g_scale=1.0):
    dev = "cuda"
    torch.manual_seed(seed)
    # q,k pre-normalised in fp32 then cast (exactly like FlashKDA's own test_fwd.py:230).
    q = torch.nn.functional.normalize(
        torch.randn(B, T, H, D, dtype=torch.float32, device=dev), p=2, dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, D, dtype=torch.float32, device=dev), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(B, T, H, D, device=dev).to(torch.bfloat16)
    g = (torch.empty(B, T, H, D, device=dev).normal_() * g_scale).to(torch.bfloat16)
    beta = torch.randn(B, T, H, device=dev).to(torch.bfloat16)
    A_log = torch.rand(H, dtype=torch.float32, device=dev)
    dt_bias = torch.rand(H, D, dtype=torch.float32, device=dev)
    # state is S^T = [N,H,V,K] for BOTH the kernel and its torch_ref (torch_ref.py:236-239).
    h0 = torch.arange(B * H * D * D, dtype=torch.float32, device=dev).reshape(B, H, D, D)
    h0 = (h0 / (B * H * D * D)).to(torch.bfloat16)
    return q, k, v, g, beta, A_log, dt_bias, h0


def run_kernel(q, k, v, g, beta, A_log, dt_bias, h0):
    import flash_kda
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros_like(q)
    fin = torch.zeros_like(h0)
    flash_kda.fwd(q, k, v, g, beta, scale, out, A_log=A_log.clone(), dt_bias=dt_bias.clone(),
                  lower_bound=LOWER_BOUND, initial_state=h0.clone(), final_state=fin)
    torch.cuda.synchronize()
    return out, fin


def run_torch_ref(q, k, v, g, beta, A_log, dt_bias, h0):
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros_like(q)
    fin = torch.zeros_like(h0)
    torch_ref(q, k, v, g, beta, scale, out, A_log=A_log, dt_bias=dt_bias,
              lower_bound=LOWER_BOUND, initial_state=h0.clone(), final_state=fin)
    return out, fin


def activated_operands(q, k, v, g, beta, A_log, dt_bias, lower_bound):
    """Reproduce exactly the pre-chunk operands FlashKDA's torch_ref builds (torch_ref.py:159-196).

    Returns the *activated* per-sequence operands in the dtype the chunk loop actually
    consumes: qn/kn bf16, g_act fp32 (log2 decay, cumsum is taken in fp32), beta_sig fp32.
    """
    H = q.shape[2]
    qn = l2_normalize_kernel_match(q)          # bf16
    kn = l2_normalize_kernel_match(k)          # bf16
    g_f = g.to(torch.float32) + dt_bias.unsqueeze(0).unsqueeze(0)   # [B,T,H,D] fp32
    a_log_exp = fp32_ex2_ftz(A_log * LOG2E).view(1, 1, H, 1)
    g_act = (lower_bound * LOG2E) * sigmoid_ext.sigmoid_tanh_fp32(a_log_exp * g_f)
    beta_sig = sigmoid_ext.sigmoid_tanh_fp32(beta.to(torch.float32))  # [B,T,H] fp32
    return qn, kn, g_act, beta_sig


def our_chunked_per_seq(qn, kn, v, g_act, beta_sig, h0, dtype):
    """Run src/kda_ref.py's chunked forward per (b,h); return o [B,T,H,D], S^T [B,H,V,K]."""
    B, T, H, _ = qn.shape
    scale = 1.0 / math.sqrt(D)
    o = torch.empty(B, T, H, D, dtype=dtype, device=qn.device)
    sT = torch.empty(B, H, D, D, dtype=dtype, device=qn.device)
    for b in range(B):
        for h in range(H):
            ob, S = kda_chunked_fwd(
                qn[b, :, h].to(dtype), kn[b, :, h].to(dtype), v[b, :, h].to(dtype),
                g_act[b, :, h].to(dtype), beta_sig[b, :, h].to(dtype),
                scale=scale, h0=h0[b, h].to(dtype).transpose(-1, -2), chunk=CHUNK)
            o[b, :, h] = ob
            sT[b, h] = S.transpose(-1, -2)      # our S is [K,V]; kernel/torch_ref store [V,K]
    return o, sT


# --------------------------------------------------------------------------------------
# Part C: an exact fp64 transcription of the torch_ref chunk loop (no bf16 quantization).
# --------------------------------------------------------------------------------------
def exact_norm(x):
    return x.double() / x.double().pow(2).sum(-1, keepdim=True).add(1e-6).sqrt()


def chunk_loop_exact_fp64(qn, kn, v, g_act, beta_sig, h0):
    """Same chunked algebra as torch_ref.py:200-244 but entirely in fp64, given the same
    already-normalised q/k and already-activated (fp64) gate/beta operands. An independent
    transcription of the blocked loop (different indexing from src/kda_ref.py)."""
    B, T, H, _ = qn.shape
    dev = qn.device
    scale = 1.0 / math.sqrt(D)
    S = h0.double().clone()                                   # [B,H,V,K] = S^T
    out = torch.empty(B, T, H, D, dtype=torch.float64, device=dev)
    for b in range(B):
        for h in range(H):
            for t0 in range(0, T, CHUNK):
                C = min(CHUNK, T - t0)
                sl = slice(t0, t0 + C)
                qc = qn[b, sl, h]; kc = kn[b, sl, h]; vc = v[b, sl, h]
                gc = g_act[b, sl, h]; bc = beta_sig[b, sl, h]
                gcum = gc.cumsum(0); gtot = gcum[-1]
                E = torch.exp2(gcum); etot = torch.exp2(gtot)
                kd = kc * E; qd = qc * E * scale
                ki = kc * torch.exp2(-gcum); kr = ki * etot[None, :]
                L = torch.tril(kd @ ki.t(), diagonal=-1) * bc[:, None]
                Mqk = torch.tril(qd @ ki.t())
                P = torch.linalg.solve(torch.eye(C, dtype=torch.float64, device=dev) + L,
                                       torch.eye(C, dtype=torch.float64, device=dev))
                Sst = S[b, h].t()                              # [K,V] view of stored S^T
                up = (vc - kd @ Sst) * bc[:, None]
                U = P @ up
                o = qd @ Sst + Mqk @ U
                Snew = kr.t() @ U + etot[:, None] * Sst
                S[b, h] = Snew.t()
                out[b, sl, h] = o
    return out, S


def case(name, B, T, H, g_scale, seed):
    print(f"\n{'='*80}\ncase [{name}]  B={B} T={T} H={H} g_scale={g_scale} seed={seed}")
    q, k, v, g, beta, A_log, dt_bias, h0 = make_inputs(B, T, H, seed, g_scale)

    # ---- Part A: kernel == FlashKDA torch_ref, bit-exact ----
    out_k, fin_k = run_kernel(q, k, v, g, beta, A_log, dt_bias, h0)
    out_r, fin_r = run_torch_ref(q, k, v, g, beta, A_log, dt_bias, h0)
    eq_o = torch.equal(out_k, out_r)
    eq_s = torch.equal(fin_k, fin_r)
    rA_o = stats("A o    kernel vs torch_ref", out_r, out_k)
    rA_s = stats("A S^T  kernel vs torch_ref", fin_r, fin_k)
    print(f"  A torch.equal(o)={eq_o}  torch.equal(S^T)={eq_s}")

    # ---- Part B: our blocked algebra fed the same activated operands, vs torch_ref ----
    qn, kn, g_act, beta_sig = activated_operands(q, k, v, g, beta, A_log, dt_bias, LOWER_BOUND)
    o_b32, s_b32 = our_chunked_per_seq(qn, kn, v, g_act, beta_sig, h0, torch.float32)
    o_b64, s_b64 = our_chunked_per_seq(qn, kn, v, g_act, beta_sig, h0, torch.float64)
    rB_o = stats("B o    our-ref(fp32) vs torch_ref", out_r, o_b32)
    rB_s = stats("B S^T  our-ref(fp32) vs torch_ref", fin_r, s_b32)
    rB_o64 = stats("B o    our-ref(fp64) vs torch_ref", out_r, o_b64)
    rB_s64 = stats("B S^T  our-ref(fp64) vs torch_ref", fin_r, s_b64)

    # ---- Part C: exact fp64 transcription vs our fp64 blocked algebra (same operands) ----
    qn_ex = exact_norm(q)
    kn_ex = exact_norm(k)
    g_act64 = g_act.double()
    beta_sig64 = beta_sig.double()
    o_exact, s_exact = chunk_loop_exact_fp64(qn_ex, kn_ex, v.double(), g_act64, beta_sig64, h0)
    o_c64, s_c64 = our_chunked_per_seq(qn_ex, kn_ex, v.double(), g_act64, beta_sig64, h0, torch.float64)
    rC_o = stats("C o    our-ref(fp64) vs exact-fp64 loop", o_exact, o_c64)
    rC_s = stats("C S^T  our-ref(fp64) vs exact-fp64 loop", s_exact, s_c64)
    # exact loop vs torch_ref (bf16): how much of the end-to-end error is bf16 quantization
    rC_k = stats("C o    exact-fp64 vs torch_ref(bf16)", out_r, o_exact)

    return dict(eq_o=eq_o, eq_s=eq_s, A_o=rA_o, A_s=rA_s, B_o=rB_o, B_s=rB_s,
                B_o64=rB_o64, B_s64=rB_s64, C_o=rC_o, C_s=rC_s, C_k=rC_k)


def main():
    cases = [
        ("B1T256H4", 1, 256, 4, 1.0, 0),
        ("B2T130H2", 2, 130, 2, 1.0, 1),      # non-multiple of 16
        ("B1T64H1", 1, 64, 1, 2.0, 2),
        ("g=0", 1, 256, 4, 0.0, 3),
        ("g=N(0,4)", 1, 192, 2, 4.0, 4),
    ]
    rows = []
    for c in cases:
        rows.append((c[0], case(*c)))

    print(f"\n{'='*80}\nSUMMARY")
    print("  Part A (kernel vs its kernel-bit-exact torch_ref):")
    for name, r in rows:
        print(f"    {name:12s} bit-exact o={r['eq_o']} S^T={r['eq_s']}  "
              f"(o max_abs={r['A_o']['max_abs']:.2e}, S max_abs={r['A_s']['max_abs']:.2e})")
    print("  Part B (our blocked algebra vs kernel-faithful torch_ref):")
    print(f"    {'case':12s} {'o.max_rel(f32)':>15s} {'S.max_rel(f32)':>15s} "
          f"{'o.max_rel(f64)':>15s} {'S.max_rel(f64)':>15s}")
    for name, r in rows:
        print(f"    {name:12s} {r['B_o']['max_rel']:15.3e} {r['B_s']['max_rel']:15.3e} "
              f"{r['B_o64']['max_rel']:15.3e} {r['B_s64']['max_rel']:15.3e}")
    print("  Part C (exact fp64 algebra):")
    for name, r in rows:
        print(f"    {name:12s} our-fp64 vs exact: o max_abs={r['C_o']['max_abs']:.2e} "
              f"S max_abs={r['C_s']['max_abs']:.2e} | exact vs torch_ref o max_rel={r['C_k']['max_rel']:.2e}")
    print("\nDONE")


if __name__ == "__main__":
    main()
