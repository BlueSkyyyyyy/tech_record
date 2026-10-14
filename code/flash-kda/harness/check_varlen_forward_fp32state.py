"""P4-5: numerically anchor the FlashKDA varlen ``StateFP32=true`` template.

P4-4 (``check_varlen_forward.py``) anchored the varlen/packed forward but only with
**bf16** initial/final state; its §7.6 explicitly left ``StateFP32=true`` (the
``flash_kda.cpp:203-207`` dispatch arm) as *待验证* ("if you want to be strict, add a
case that passes fp32 ``initial_state``/``final_state``").

This harness closes that gap and, while doing so, answers a "why" that is easy to get
wrong: does ``StateFP32`` actually give a **more precise** running state? Reading
``fwd_kernel2.cuh`` says **no** -- the recurrence state lives in ``state_acc``, a
``BF16`` smem array (``fwd_kernel2.cuh:82``); ``StateFP32`` only changes the *I/O shell*:
it TMA-loads an fp32 ``initial_state`` into a staging buffer and rounds it to bf16
(``:300-303``), and at the end rounds the bf16 state back up to fp32 (``:811-814``).
So the whole recurrence is still bf16. This harness turns that structural reading into
measured facts.

Parts
-----
A. **Shell equivalence (bit-exact)** -- varlen with fp32 h0/final vs varlen with
   ``bf16(h0)`` / bf16 final (same fp32 operand master). Predicted: ``o`` bit-identical,
   fp32 final == exact upcast of the bf16 final. Also the isolated (per-sequence
   ``B=1``) vs packed variant, mirroring P4-4 Part A.
B. **vs fp64 golden** -- both the fp32-state and bf16-state varlen runs vs the fp64
   token-recurrent golden; the fp32-state one must land at the *same* bf16 floor (no
   precision gain) -- the quantitative form of "StateFP32 is an I/O shell".
C. **Genuine fp32 h0** -- feed an h0 that is *not* bf16-representable; confirm the
   fp32-state path rounds it exactly like ``h0.to(bf16)`` (the extra mantissa is
   discarded internally), so the only thing ``StateFP32`` buys is that the caller need
   not cast to bf16 explicitly.
D. **State-combination sweep** -- fp32 in-only / out-only / both, each vs ``B=1``.

Run inside `kernel_lab` (needs the built ``flash_kda_C``):
    python ~/proj/tech_record/code/flash-kda/harness/check_varlen_forward_fp32state.py
"""

import os
import math
import sys

import torch

SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, SRC)
from kda_ref import kda_gate, kda_token_recurrent, l2_normalize  # noqa: E402

CSRC = os.path.expanduser("~/github/FlashKDA")  # only for the module path fallback
sys.path.insert(0, CSRC)

D = 128
CHUNK = 16


def make_inputs(lens, H, g_scale, seed, fp32_h0=False):
    """Return packed [1, T_total, H, D] operands + fp32 masters, per-seq h0 (fp32)."""
    dev = "cuda"
    torch.manual_seed(seed)
    T_total = sum(lens)
    N = len(lens)
    q = torch.randn(1, T_total, H, D, device=dev)
    k = torch.randn(1, T_total, H, D, device=dev)
    v = torch.randn(1, T_total, H, D, device=dev)
    g = torch.randn(1, T_total, H, D, device=dev) * g_scale
    beta = torch.randn(1, T_total, H, device=dev)
    A_log = torch.randn(H, device=dev)
    dt_bias = torch.randn(H, D, device=dev)
    h0 = torch.randn(N, H, D, D, device=dev)
    if not fp32_h0:
        # make h0 exactly bf16-representable -> isolates the "shell" question
        h0 = h0.bfloat16().float()
    return dict(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias,
                h0=h0, lens=lens, N=N, H=H, T_total=T_total)


def cu_from_lens(lens):
    return torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()),
                        dtype=torch.int64, device="cuda")


def run_varlen(x, lower_bound=-5.0, with_in=True, with_out=True, state_fp32=True):
    from flash_kda_C import fwd as fwd_raw, get_workspace_size

    q, k, v, g, beta = x["q"], x["k"], x["v"], x["g"], x["beta"]
    H, N, T_total = x["H"], x["N"], x["T_total"]
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros(1, T_total, H, D, dtype=torch.bfloat16, device="cuda")
    ws = torch.empty(int(get_workspace_size(T_total, H, N)), dtype=torch.uint8, device="cuda")
    st_dtype = torch.float32 if state_fp32 else torch.bfloat16
    fin = torch.zeros(N, H, D, D, dtype=st_dtype, device="cuda")
    # internal layout is S^T = [N,H,V,K]
    init = x["h0"].transpose(-1, -2).contiguous().to(st_dtype)
    fwd_raw(q.bfloat16().contiguous(), k.bfloat16().contiguous(), v.bfloat16().contiguous(),
            g.bfloat16().contiguous(), beta.bfloat16().contiguous(), scale, out, ws,
            x["A_log"].clone(), x["dt_bias"].clone(), lower_bound,
            initial_state=init if with_in else None,
            final_state=fin if with_out else None,
            cu_seqlens=cu_from_lens(x["lens"]))
    torch.cuda.synchronize()
    return out, (fin if with_out else None)


def run_batched_single(x, i, lower_bound=-5.0, with_in=True, with_out=True, state_fp32=True):
    """Run sequence ``i`` on its own through the ordinary (non-varlen) path, B=1."""
    from flash_kda_C import fwd as fwd_raw, get_workspace_size

    H = x["H"]
    a, b = x["cu"][i].item(), x["cu"][i + 1].item()
    scale = 1.0 / math.sqrt(D)
    sl = slice(a, b)
    L = b - a
    q = x["q"][:, sl].contiguous()
    k = x["k"][:, sl].contiguous()
    v = x["v"][:, sl].contiguous()
    g = x["g"][:, sl].contiguous()
    beta = x["beta"][:, sl].contiguous()
    out = torch.zeros(1, L, H, D, dtype=torch.bfloat16, device="cuda")
    ws = torch.empty(int(get_workspace_size(L, H, 1)), dtype=torch.uint8, device="cuda")
    st_dtype = torch.float32 if state_fp32 else torch.bfloat16
    fin = torch.zeros(1, H, D, D, dtype=st_dtype, device="cuda")
    init = x["h0"][i:i + 1].transpose(-1, -2).contiguous().to(st_dtype)
    fwd_raw(q.bfloat16(), k.bfloat16(), v.bfloat16(), g.bfloat16(), beta.bfloat16(),
            scale, out, ws, x["A_log"].clone(), x["dt_bias"].clone(), lower_bound,
            initial_state=init if with_in else None,
            final_state=fin if with_out else None)
    torch.cuda.synchronize()
    return out, (fin if with_out else None)


def golden(x, i, lower_bound=-5.0):
    H = x["H"]
    a, b = x["cu"][i].item(), x["cu"][i + 1].item()
    scale = 1.0 / math.sqrt(D)
    sl = slice(a, b)
    g_act = kda_gate(x["g"][0, sl].double(), x["A_log"].double(),
                     x["dt_bias"].double(), lower_bound)
    o = torch.empty(b - a, H, D, dtype=torch.float64, device="cuda")
    s = torch.empty(H, D, D, dtype=torch.float64, device="cuda")
    for h in range(H):
        qn = l2_normalize(x["q"][0, sl, h].double())
        kn = l2_normalize(x["k"][0, sl, h].double())
        oi, si = kda_token_recurrent(
            qn, kn, x["v"][0, sl, h].double(), g_act[:, h],
            torch.sigmoid(x["beta"][0, sl, h].double()), scale, x["h0"][i, h].double())
        o[:, h] = oi.float()
        s[h] = si.float()
    return o, s


def main():
    print("torch", torch.__version__)
    from flash_kda_C import fwd as _  # noqa: F401
    print("flash_kda_C imported OK")

    cases = [
        ("lens=[30,16,5,50]", [30, 16, 5, 50], 2, 2.0, 0),
        ("lens=[1,16,17,64,3]", [1, 16, 17, 64, 3], 4, 2.0, 1),
        ("lens=[64,64,64]", [64, 64, 64], 2, 2.0, 2),
        ("lens=[250,7] T=257", [250, 7], 2, 2.0, 3),
        ("lens=[33,47,19,100] T=199", [33, 47, 19, 100], 3, 2.0, 4),
        ("g=N(0,0) no decay", [30, 16, 5, 50], 2, 0.0, 5),
        ("g=N(0,4) strong decay", [30, 16, 5, 50], 2, 4.0, 6),
    ]

    # ------------------------------------------------------------------ Part A
    print("\n" + "=" * 78)
    print("Part A: varlen fp32-state  vs  bf16-state (h0 = bf16-representable)")
    print("        predicts o BIT-IDENTICAL + fp32 final == upcast(bf16 final)")
    print("=" * 78)
    for name, lens, H, gs, seed in cases:
        x = make_inputs(lens, H, gs, seed, fp32_h0=False)
        x["cu"] = cu_from_lens(lens)
        out_f, fin_f = run_varlen(x, state_fp32=True)
        out_b, fin_b = run_varlen(x, state_fp32=False)
        do = (out_f.float() - out_b.float()).abs().max().item()
        ds = (fin_f - fin_b.float()).abs().max().item()
        # per-sequence batched, fp32 shell
        mo = ms = 0.0
        for i in range(x["N"]):
            a, b = x["cu"][i].item(), x["cu"][i + 1].item()
            out_i, fin_i = run_batched_single(x, i, state_fp32=True)
            mo = max(mo, (out_f[0, a:b].float() - out_i[0].float()).abs().max().item())
            ms = max(ms, (fin_f[i] - fin_i[0]).abs().max().item())
        print(f"  {name:28s} o(f32 vs bf16 shell)={do:.3e}  "
              f"S(f32 vs upcast bf16)={ds:.3e}  |  packed-vs-B1 o={mo:.3e} S={ms:.3e}")
    print("  (packed == per-seq batched bit-identical; f32 shell == bf16 shell for bf16 h0)")

    # ------------------------------------------------------------------ Part B
    print("\n" + "=" * 78)
    print("Part B: varlen vs fp64 token-recurrent golden -- fp32-state vs bf16-state")
    print("=" * 78)
    for name, lens, H, gs, seed in cases:
        x = make_inputs(lens, H, gs, seed, fp32_h0=True)  # genuine fp32 h0
        x["cu"] = cu_from_lens(lens)
        out_f, fin_f = run_varlen(x, state_fp32=True)
        out_b, fin_b = run_varlen(x, state_fp32=False)
        rof = rosf = rob = rosb = 0.0
        for i in range(x["N"]):
            a, b = x["cu"][i].item(), x["cu"][i + 1].item()
            o_g, s_g = golden(x, i)
            og = o_g.float().abs().max().item() + 1e-12
            sg = s_g.float().abs().max().item() + 1e-12
            rof = max(rof, (o_g.float() - out_f[0, a:b].float()).abs().max().item() / og)
            rob = max(rob, (o_g.float() - out_b[0, a:b].float()).abs().max().item() / og)
            rosf = max(rosf, (s_g.transpose(-1, -2).float() - fin_f[i]).abs().max().item() / sg)
            rosb = max(rosb, (s_g.transpose(-1, -2).float() - fin_b[i].float()).abs().max().item() / sg)
        print(f"  {name:28s} fp32-state o={rof:.3e} S={rosf:.3e} | bf16-state o={rob:.3e} S={rosb:.3e}")

    # ------------------------------------------------------------------ Part C
    print("\n" + "=" * 78)
    print("Part C: genuine fp32 h0 is rounded to bf16 internally -- identical to h0.to(bf16)")
    print("=" * 78)
    name, lens, H, gs, seed = cases[0]
    xg = make_inputs(lens, H, gs, seed, fp32_h0=True)          # genuine fp32 h0
    xg["cu"] = cu_from_lens(lens)
    out_g, fin_g = run_varlen(xg, state_fp32=True)
    # same operands but feed the *bf16-rounded* h0 through the bf16 shell
    xb = dict(xg)
    xb["h0"] = xg["h0"].bfloat16().float()
    out_b, fin_b = run_varlen(xb, state_fp32=False)
    d_round = (xg["h0"].transpose(-1, -2).float() - xg["h0"].transpose(-1, -2).bfloat16().float()).abs().max().item()
    print(f"  |h0_fp32 - round_bf16(h0_fp32)|      = {d_round:.3e}")
    print(f"  o(fp32-state, genuine h0) vs o(bf16 shell, bf16(h0)) = "
          f"{(out_g.float() - out_b.float()).abs().max().item():.3e}")
    print(f"  S(fp32-state, genuine h0) vs S(bf16 shell, bf16(h0)) = "
          f"{(fin_g - fin_b.float()).abs().max().item():.3e}")

    # ------------------------------------------------------------------ Part D
    print("\n" + "=" * 78)
    print("Part D: fp32-state-combination sweep on the varlen template (lens=[30,16,5,50])")
    print("=" * 78)
    x = make_inputs(cases[0][1], cases[0][2], cases[0][3], cases[0][4], fp32_h0=False)
    x["cu"] = cu_from_lens(x["lens"])
    combos = [("no state", False, False), ("state-in only (fp32)", True, False),
              ("state-out only (fp32)", False, True), ("both (fp32)", True, True)]
    for tag, wi, wo in combos:
        out_v, fin_v = run_varlen(x, with_in=wi, with_out=wo, state_fp32=True)
        mo = ms = 0.0
        for i in range(x["N"]):
            a, b = x["cu"][i].item(), x["cu"][i + 1].item()
            out_i, fin_i = run_batched_single(x, i, with_in=wi, with_out=wo, state_fp32=True)
            mo = max(mo, (out_v[0, a:b].float() - out_i[0].float()).abs().max().item())
            if wo:
                ms = max(ms, (fin_v[i] - fin_i[0]).abs().max().item())
        print(f"  {tag:22s} o vs per-seq batched max_abs={mo:.3e}  "
              f"S max_abs={ms:.3e}" + ("" if wo else "  (no out)"))

    print("\nDONE")


if __name__ == "__main__":
    with torch.no_grad():
        main()
