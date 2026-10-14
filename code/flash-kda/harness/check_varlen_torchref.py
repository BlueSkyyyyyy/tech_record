"""P4-7: intersect the two most recent forward threads — varlen (P4-4/P4-5) and the
kernel-bit-exact oracle (P4-6).

Background
----------
P4-4 anchored the packed/``cu_seqlens`` forward only against a *per-sequence batched*
kernel run (bit-exact) and an fp64 token-recurrent golden (bf16 floor). P4-6 introduced
FlashKDA's own ``tests/torch_ref.py`` — a pure-PyTorch re-implementation that mirrors the
kernel **bit-for-bit** — but only ever ran it on the *batched* path (equal-length ``B``).

``torch_ref`` itself has a dedicated ``cu_seqlens`` branch (``torch_ref.py:141-193``:
``B must be 1``, per-sequence ``work_state[seq_idx]``, chunk loop bounded by each
sequence's ``[bos, eos)``). That branch had **never been exercised** in this series, nor
had the packed kernel been compared to a bit-exact oracle (only to another kernel run).

This harness closes the intersection:

A. **Packed kernel vs ``torch_ref(cu_seqlens=...)``** on the *same* packed operands and
   *distinct per-sequence* ``initial_state`` — require ``o`` and ``final_state`` to be
   **bit-identical** (``max_abs=0``). This is a strictly stronger anchor than P4-4 (which
   compared the packed kernel to *another kernel run*): it validates the packed path
   against the kernel's *algebra* as transcribed independently in ``torch_ref``, and it
   exercises the reference's own ``cu_seqlens`` branch end-to-end.
B. **Reference packed vs reference per-sequence** — run the same sequences through
   ``torch_ref`` one at a time (``B=1``, no ``cu_seqlens``) and require per-sequence
   bit-equality. This isolates the reference's own packing loop (independent of the
   kernel), the analogue of P4-4 Part A but for the oracle.

Together A and B form a bit-exact triangle ``packed-kernel == ref-packed == ref-per-seq``.

Cases mirror P4-4: ``L=1<CHUNK``, ``L=CHUNK``, non-divisible total ``T``, ``g=0``,

Run inside `kernel_lab` (needs the built ``flash_kda_C`` + nvcc for the inline sigmoid):
    CUDA_VISIBLE_DEVICES=<idle> python ~/proj/tech_record/code/flash-kda/harness/check_varlen_torchref.py
"""

import os
import math
import sys

import torch

SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, SRC)  # noqa: E402

FLASHKDA = os.path.expanduser("~/github/FlashKDA")
sys.path.insert(0, FLASHKDA)
sys.path.insert(0, f"{FLASHKDA}/tests")
from torch_ref import torch_ref  # noqa: E402

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


def make_inputs(lens, H, g_scale, seed):
    """Packed [1, T_total, H, D] operands + per-seq h0 (S^T layout [N,H,V,K])."""
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
    return dict(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias,
                h0=h0, lens=lens, N=N, H=H, T_total=T_total)


def cu_from_lens(lens):
    return torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()),
                        dtype=torch.int64, device="cuda")


def run_kernel_varlen(x):
    """FlashKDA packed forward (B=1) with per-sequence initial_state, bf16 in/out."""
    from flash_kda_C import fwd as fwd_raw, get_workspace_size

    q, k, v, g, beta = x["q"], x["k"], x["v"], x["g"], x["beta"]
    H, N, T_total = x["H"], x["N"], x["T_total"]
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros(1, T_total, H, D, dtype=torch.bfloat16, device="cuda")
    ws = torch.empty(int(get_workspace_size(T_total, H, N)), dtype=torch.uint8, device="cuda")
    fin = torch.zeros(N, H, D, D, dtype=torch.bfloat16, device="cuda")
    init = x["h0"].transpose(-1, -2).contiguous().bfloat16()  # [N,H,V,K] = S^T
    fwd_raw(q.bfloat16().contiguous(), k.bfloat16().contiguous(), v.bfloat16().contiguous(),
            g.bfloat16().contiguous(), beta.bfloat16().contiguous(), scale, out, ws,
            x["A_log"].clone(), x["dt_bias"].clone(), LOWER_BOUND,
            initial_state=init, final_state=fin,
            cu_seqlens=cu_from_lens(x["lens"]))
    torch.cuda.synchronize()
    return out, fin


def run_ref_packed(x):
    """torch_ref on the whole packed stream via its cu_seqlens branch (B must be 1)."""
    q, k, v, g, beta = x["q"], x["k"], x["v"], x["g"], x["beta"]
    H, N, T_total = x["H"], x["N"], x["T_total"]
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros(1, T_total, H, D, dtype=torch.bfloat16, device="cuda")
    fin = torch.zeros(N, H, D, D, dtype=torch.bfloat16, device="cuda")
    init = x["h0"].transpose(-1, -2).contiguous().bfloat16()  # [N,H,V,K] = S^T
    torch_ref(q.bfloat16().contiguous(), k.bfloat16().contiguous(), v.bfloat16().contiguous(),
              g.bfloat16().contiguous(), beta.bfloat16().contiguous(), scale, out,
              A_log=x["A_log"].clone(), dt_bias=x["dt_bias"].clone(), lower_bound=LOWER_BOUND,
              initial_state=init, final_state=fin, cu_seqlens=cu_from_lens(x["lens"]))
    torch.cuda.synchronize()
    return out, fin


def run_ref_single(x, i):
    """torch_ref on one sequence (B=1, no cu_seqlens -> implicit [0,L]); returns S^T."""
    H = x["H"]
    cu = cu_from_lens(x["lens"])
    a, b = cu[i].item(), cu[i + 1].item()
    L = b - a
    scale = 1.0 / math.sqrt(D)
    sl = slice(a, b)
    out = torch.zeros(1, L, H, D, dtype=torch.bfloat16, device="cuda")
    fin = torch.zeros(1, H, D, D, dtype=torch.bfloat16, device="cuda")
    init = x["h0"][i:i + 1].transpose(-1, -2).contiguous().bfloat16()
    torch_ref(x["q"][:, sl].contiguous().bfloat16(), x["k"][:, sl].contiguous().bfloat16(),
              x["v"][:, sl].contiguous().bfloat16(), x["g"][:, sl].contiguous().bfloat16(),
              x["beta"][:, sl].contiguous().bfloat16(), scale, out,
              A_log=x["A_log"].clone(), dt_bias=x["dt_bias"].clone(), lower_bound=LOWER_BOUND,
              initial_state=init, final_state=fin)
    torch.cuda.synchronize()
    return out, fin


CASES = [
    ("lens=[30,16,5,50]", [30, 16, 5, 50], 2, 2.0, 0),
    ("lens=[1,16,17,64,3]", [1, 16, 17, 64, 3], 4, 2.0, 1),   # L=1 < CHUNK
    ("lens=[64,64,64] mult 16", [64, 64, 64], 2, 2.0, 2),
    ("lens=[250,7] non-div T=257", [250, 7], 2, 2.0, 3),
    ("lens=[33,47,19,100] T=199", [33, 47, 19, 100], 3, 2.0, 4),
    ("g=0 no decay", [30, 16, 5, 50], 2, 0.0, 5),
    ("g=N(0,4) strong decay", [30, 16, 5, 50], 2, 4.0, 6),
]


def main():
    print("torch", torch.__version__)
    from flash_kda_C import fwd as _  # noqa: F401
    print("flash_kda_C imported OK")

    print("\n" + "=" * 82)
    print("Part A: packed kernel  vs  torch_ref(cu_seqlens)   expect BIT-IDENTICAL")
    print("=" * 82)
    a_rows = []
    for name, lens, H, gs, seed in CASES:
        x = make_inputs(lens, H, gs, seed)
        out_k, fin_k = run_kernel_varlen(x)
        out_r, fin_r = run_ref_packed(x)
        eq_o = torch.equal(out_k, out_r)
        eq_s = torch.equal(fin_k, fin_r)
        ra_o = stats("A o   kernel vs torch_ref", out_r, out_k)
        ra_s = stats("A S^T kernel vs torch_ref", fin_r, fin_k)
        print(f"    => torch.equal(o)={eq_o}  torch.equal(S^T)={eq_s}")
        a_rows.append((name, eq_o, eq_s, ra_o["max_abs"], ra_s["max_abs"]))

    print("\n" + "=" * 82)
    print("Part B: torch_ref packed  vs  torch_ref per-sequence   expect BIT-IDENTICAL")
    print("=" * 82)
    b_rows = []
    for name, lens, H, gs, seed in CASES:
        x = make_inputs(lens, H, gs, seed)
        cu = cu_from_lens(lens)
        out_p, fin_p = run_ref_packed(x)
        mo = ms = 0.0
        for i in range(x["N"]):
            a, b = cu[i].item(), cu[i + 1].item()
            out_i, fin_i = run_ref_single(x, i)
            do = (out_p[0, a:b].float() - out_i[0].float()).abs().max().item()
            ds = (fin_p[i].float() - fin_i[0].float()).abs().max().item()
            mo = max(mo, do)
            ms = max(ms, ds)
        print(f"  {name:28s} T_total={x['T_total']:4d} N={x['N']}  "
              f"o max_abs={mo:.3e}  S^T max_abs={ms:.3e}")
        b_rows.append((name, mo, ms))

    print("\n" + "=" * 82)
    print("SUMMARY")
    print("=" * 82)
    print("  Part A (packed kernel vs kernel-bit-exact torch_ref @ cu_seqlens):")
    for name, eq_o, eq_s, ao, as_ in a_rows:
        print(f"    {name:28s} bit-exact o={eq_o} S^T={eq_s}  "
              f"(o max_abs={ao:.2e}, S max_abs={as_:.2e})")
    allA = all(r[1] and r[2] for r in a_rows)
    allB = all(r[1] == 0.0 and r[2] == 0.0 for r in b_rows)
    print(f"  Part A all bit-exact: {allA}")
    print(f"  Part B all bit-exact: {allB}")
    print("\nDONE")


if __name__ == "__main__":
    with torch.no_grad():
        main()
