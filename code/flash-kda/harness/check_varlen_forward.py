"""P4-4: numerically anchor the FlashKDA *varlen* (packed ``cu_seqlens``) forward.

Prior numeric work (P4-2/P4-3, ``check_fwd_numerics.py`` / ``check_three_way.py``) only
ever ran the **batched** path (``B`` sequences of equal length ``T``). The varlen path
is exercised elsewhere only for *bookkeeping* (``check_workspace_layout.py`` Part C: the
``tile_prefix`` int32 array lands at the predicted offset). This harness closes the gap:
does packing multiple sequences with ``cu_seqlens`` produce, per sequence, exactly the
same output/state as running each sequence on its own?

Parts
-----
A. **Isolation / determinism (bit-exact)** — run the packed varlen forward, then run every
   sequence separately through the ordinary batched path (``B=1``) with the *same* inputs
   and the *same* per-sequence ``initial_state``; require the per-sequence output slice and
   ``final_state`` row to be **bit-identical** (``max_abs=0``). This is the real isolation
   test: any tile-prefix / binary-search / off-by-one in the tile→sequence mapping shows up
   here as O(1) differences.
B. **vs fp64 golden** — each sequence slice vs the pure-PyTorch token-recurrent fp64 golden
   (``src/kda_ref.py``), expected to land at the bf16 floor.
C. **Per-sequence initial state is honoured** — with distinct ``h0_i`` per sequence, compare
   the packed result against a *shared-h0* run; the difference must equal the isolated
   per-sequence difference (i.e. sequence ``i`` really consumes ``h0_i``, not a shared one).
D. **Edge lengths** — ``L=1`` (shorter than CHUNK), ``L=CHUNK``, lengths not multiples of
   CHUNK, and a total ``T_total`` not divisible by CHUNK.
E. **State combinations** — `HasStateIn/HasStateOut` in the varlen template must each be
   handled: no-state, state-in only, state-out only, both (bf16).

Run inside `kernel_lab` (needs the built ``flash_kda_C``):
    python ~/proj/tech_record/code/flash-kda/harness/check_varlen_forward.py
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


def stats(label, gold, pred):
    d = (gold.float() - pred.float()).abs()
    ref = gold.float().abs()
    max_abs = d.max().item()
    max_rel = (d.max() / (ref.max() + 1e-12)).item()
    print(f"  {label:34s} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")
    return dict(max_abs=max_abs, max_rel=max_rel)


def make_inputs(lens, H, g_scale, seed):
    """Return packed [1, T_total, H, D] bf16 operands + fp32 masters, per-seq h0."""
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


def run_varlen(x, lower_bound=-5.0, with_in=True, with_out=True):
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
            x["A_log"].clone(), x["dt_bias"].clone(), lower_bound,
            initial_state=init if with_in else None,
            final_state=fin if with_out else None,
            cu_seqlens=cu_from_lens(x["lens"]))
    torch.cuda.synchronize()
    return out, (fin if with_out else None)


def run_batched_single(x, i, lower_bound=-5.0, with_in=True, with_out=True):
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
    fin = torch.zeros(1, H, D, D, dtype=torch.bfloat16, device="cuda")
    init = x["h0"][i:i + 1].transpose(-1, -2).contiguous().bfloat16()
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

    # ---- Part A/B/D: isolation (bit-exact) + golden, several length sets & regimes ----
    cases = [
        ("lens=[30,16,5,50]", [30, 16, 5, 50], 2, 2.0, 0),
        ("lens=[1,16,17,64,3]", [1, 16, 17, 64, 3], 4, 2.0, 1),
        ("lens=[64,64,64] (mult of 16)", [64, 64, 64], 2, 2.0, 2),
        ("lens=[250,7] non-div T=257", [250, 7], 2, 2.0, 3),
        ("lens=[33,47,19,100] T=199", [33, 47, 19, 100], 3, 2.0, 4),
        ("g=N(0,0) no decay", [30, 16, 5, 50], 2, 0.0, 5),
        ("g=N(0,4) strong decay", [30, 16, 5, 50], 2, 4.0, 6),
    ]

    print("\n" + "=" * 78)
    print("Part A: packed varlen  vs  per-sequence batched (B=1)   expect BIT-IDENTICAL")
    print("=" * 78)
    a_rows = []
    for name, lens, H, gs, seed in cases:
        x = make_inputs(lens, H, gs, seed)
        x["cu"] = cu_from_lens(lens)
        out_v, fin_v = run_varlen(x)
        mo = ms = 0.0
        for i in range(x["N"]):
            a, b = x["cu"][i].item(), x["cu"][i + 1].item()
            out_i, fin_i = run_batched_single(x, i)
            do = (out_v[0, a:b].float() - out_i[0].float()).abs().max().item()
            ds = (fin_v[i].float() - fin_i[0].float()).abs().max().item()
            mo = max(mo, do)
            ms = max(ms, ds)
        print(f"  {name:30s} T_total={x['T_total']:4d} N={x['N']}  "
              f"o max_abs={mo:.3e}  S max_abs={ms:.3e}")
        a_rows.append((name, mo, ms))

    print("\n" + "=" * 78)
    print("Part B: packed varlen  vs  fp64 token-recurrent golden")
    print("=" * 78)
    b_rows = []
    for name, lens, H, gs, seed in cases:
        x = make_inputs(lens, H, gs, seed)
        x["cu"] = cu_from_lens(lens)
        out_v, fin_v = run_varlen(x)
        ro = rs = 0.0
        for i in range(x["N"]):
            a, b = x["cu"][i].item(), x["cu"][i + 1].item()
            o_g, s_g = golden(x, i)
            do = (o_g.float() - out_v[0, a:b].float()).abs().max().item()
            dg = o_g.float().abs().max().item() + 1e-12
            ds = (s_g.transpose(-1, -2).float() - fin_v[i].float()).abs().max().item()
            sg = s_g.float().abs().max().item() + 1e-12
            ro = max(ro, do / dg)
            rs = max(rs, ds / sg)
        print(f"  {name:30s} o max_rel={ro:.3e}  S^T max_rel={rs:.3e}")
        b_rows.append((name, ro, rs))

    print("\n" + "=" * 78)
    print("Part C: per-sequence h0 is honoured (packed distinct-h0 vs shared-h0)")
    print("=" * 78)
    lens, H, gs, seed = [30, 16, 5, 50], 2, 2.0, 0
    x = make_inputs(lens, H, gs, seed)
    x["cu"] = cu_from_lens(lens)
    out_d, fin_d = run_varlen(x)
    # shared h0: broadcast sequence 0's state to all sequences
    x_shared = dict(x)
    x_shared["h0"] = x["h0"][0:1].repeat(x["N"], 1, 1, 1)
    out_s, fin_s = run_varlen(x_shared)
    for i in range(x["N"]):
        # isolated per-sequence difference (distinct vs shared h0), same slice index i
        a, b = x["cu"][i].item(), x["cu"][i + 1].item()
        oi_d, _ = run_batched_single(x, i)          # h0 = h0_i
        oi_s, _ = run_batched_single(x_shared, i)    # h0 = h0_0
        iso = (oi_d[0].float() - oi_s[0].float()).abs().max().item()
        pac = (out_d[0, a:b].float() - out_s[0, a:b].float()).abs().max().item()
        print(f"  seq {i} (len {b - a:3d}): isolated |o(h0_i)-o(h0_0)|={iso:.3e}  "
              f"packed same={pac:.3e}  (bit-identical: {iso == pac})")
    print(f"  seq0 final-state identical under both: max_abs="
          f"{(fin_d[0].float() - fin_s[0].float()).abs().max().item():.3e}  "
          f"(seq0 h0 unchanged); seq1 changed by "
          f"{(fin_d[1].float() - fin_s[1].float()).abs().max().item():.3e}")

    print("\n" + "=" * 78)
    print("Part E: state-combination sweep on the varlen template (lens=[30,16,5,50])")
    print("=" * 78)
    x = make_inputs(lens, H, 2.0, 0)
    x["cu"] = cu_from_lens(lens)
    combos = [("no state", False, False), ("state-in only", True, False),
              ("state-out only", False, True), ("both (bf16)", True, True)]
    for tag, wi, wo in combos:
        out_v, fin_v = run_varlen(x, with_in=wi, with_out=wo)
        mo = ms = 0.0
        for i in range(x["N"]):
            a, b = x["cu"][i].item(), x["cu"][i + 1].item()
            out_i, fin_i = run_batched_single(x, i, with_in=wi, with_out=wo)
            mo = max(mo, (out_v[0, a:b].float() - out_i[0].float()).abs().max().item())
            if wo:
                ms = max(ms, (fin_v[i].float() - fin_i[0].float()).abs().max().item())
        print(f"  {tag:16s} o vs per-seq batched max_abs={mo:.3e}  "
              f"S max_abs={ms:.3e}" + ("" if wo else "  (no out)"))

    print("\nDONE")


if __name__ == "__main__":
    with torch.no_grad():
        main()
