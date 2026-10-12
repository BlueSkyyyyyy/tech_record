"""P6-1 evidence: map FLA `chunk_kda` backward kernels <-> the chunked-backward symbols
of `docs/05` §7, and profile the backward kernel decomposition.

Part A (numerical): run FLA's *forward sub-functions* on fp32 input and compare the
intermediate tensors it saves for backward (`qg, kg, Aqk, Akk, w, v_new, h`) against the
chunked-forward symbols of `src/kda_ref.py::kda_chunked_fwd`
(`qd, kd, Mqk, P, U, S_in`).  This pins down the notation used by the backward kernels.

Part B (structural): under `torch.profiler`, run a real FLA `chunk_kda` forward+backward and
list the Triton kernel launch counts, i.e. the actual kernel decomposition.

Run inside `kernel_lab`:
    python ~/proj/tech_record/code/flash-kda/harness/check_bwd_kernel_map.py
"""
import math
import sys

import torch

sys.path.insert(0, "~/proj/tech_record/code/flash-kda/src")

RCP_LN2 = 1.4426950408889634  # log2(e); FLA uses it to turn natural-log decay into log2


def stats(name, a, b):
    d = (a.float() - b.float()).abs()
    ref = b.float().abs().max().item()
    max_abs = d.max().item()
    max_rel = max_abs / (ref + 1e-12)
    print(f"  {name:28s} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")
    return max_abs, max_rel


def fla_forward_intermediates(q, k, v, g_log2, beta, h0, scale, BT):
    """Reproduce FLA's forward pipeline and return the tensors it saves for backward.

    Naming follows FLA (`chunk.py:74`): (o, final_state, g, Aqk, Akk, w, u, qg, kg, v_new, h).
    Layout: [B, T, H, K] / [B, HV, K, V]; input `g_log2` is our log2 decay.
    """
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
    from fla.ops.utils import chunk_local_cumsum

    g_fla = g_log2 * math.log(2.0)  # natural-log decay, FLA convention
    g_cum = chunk_local_cumsum(g_fla, scale=RCP_LN2, chunk_size=BT)  # log2 cumulative
    # disable_recompute=True -> keep qg/kg (the decayed q/k the backward kernels consume)
    w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q=q, k=k, v=v, gk=g_cum, beta=beta, scale=scale, chunk_size=BT,
        disable_recompute=True)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g_cum, initial_state=h0, output_final_state=True, chunk_size=BT)
    return dict(g_cum=g_cum, w=w, u=u, qg=qg, kg=kg, Aqk=Aqk, Akk=Akk, v_new=v_new, h=h)


def inv_lower(L):
    """Stable (I+L)^-1 for strictly-lower L via forward substitution (matches the kernel;
    a dense fp32 `torch.linalg.solve` on an ill-conditioned unit-triangular matrix loses
    all precision here and even corrupts the trivial first row)."""
    C = L.shape[0]
    P = torch.zeros_like(L)
    for i in range(C):
        row = torch.zeros(C, dtype=L.dtype, device=L.device)
        if i > 0:
            row = -(L[i, :i, None] * P[:i]).sum(0)   # length-C vector
        row[i] = 1.0
        P[i] = row
    return P


def ref_forward_intermediates(q, k, v, g_log2, beta, h0, scale, BT):
    """Chunked-forward intermediates of docs/05 §7 (single head, [T,K]) for one head."""
    T, D = q.shape
    states_in = []
    qd_all, kd_all, Mqk_all, P_all, U_all, kr_all = [], [], [], [], [], []
    S = torch.zeros(D, D, dtype=q.dtype, device=q.device) if h0 is None else h0.clone()
    for t0 in range(0, T, BT):
        C = min(BT, T - t0)
        sl = slice(t0, t0 + C)
        qc, kc, vc, gc, bc = q[sl], k[sl], v[sl], g_log2[sl], beta[sl]
        if C < BT:
            pad = BT - C
            z = lambda x: torch.cat([x, x.new_zeros(pad, *x.shape[1:])], 0)
            qc, kc, vc, gc = z(qc), z(kc), z(vc), z(gc)
            bc = torch.cat([bc, bc.new_zeros(pad)], 0)
        gcum = gc.cumsum(0)
        gtot = gcum[-1]
        E = torch.exp2(gcum)
        etot = torch.exp2(gtot)
        kd = kc * E
        qd = qc * E * scale
        ki = kc * torch.exp2(-gcum)
        kr = ki * etot[None, :]
        L = torch.tril(kd @ ki.T, -1) * bc[:, None]
        Mqk = torch.tril(qd @ ki.T)
        P = inv_lower(L)
        uprime = (vc - kd @ S) * bc[:, None]
        U = P @ uprime
        states_in.append(S.clone())
        qd_all.append(qd); kd_all.append(kd); Mqk_all.append(Mqk); P_all.append(P)
        U_all.append(U); kr_all.append(kr)
        S = kr.T @ U + etot[:, None] * S
    st = lambda xs: torch.stack(xs, 0)
    return dict(kd=st(kd_all), qd=st(qd_all), Mqk=st(Mqk_all), P=st(P_all),
                U=st(U_all), kr=st(kr_all), S_in=st(states_in)), S


def part_a(device):
    print("=" * 78)
    print("Part A: FLA saved-for-backward tensors  vs  docs/05 §7 chunked symbols (fp32)")
    print("=" * 78)
    torch.manual_seed(0)
    B, T, H, K, V = 1, 128, 2, 64, 64
    BT = 32  # FLA chunk_size is restricted to 32/64
    scale = K ** -0.5
    # real KDA L2-normalises q,k; with unnormalised keys |k_i.k_j| ~ sqrt(K) the
    # within-chunk inverse is ill-conditioned and the fp32 states blow up.
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    v = torch.randn(B, T, H, V, device=device)
    g_log2 = -torch.rand(B, T, H, K, device=device) * 0.5
    beta = torch.sigmoid(torch.randn(B, T, H, device=device))
    h0 = torch.randn(B, H, K, V, device=device) * 0.1

    fi = fla_forward_intermediates(q, k, v, g_log2, beta, h0, scale, BT)
    print(f"  config: B={B} T={T} H={H} K={K} V={V} BT={BT} dtype={q.dtype}")

    worst = 0.0
    for h in range(H):
        ri, _ = ref_forward_intermediates(
            q[0, :, h], k[0, :, h], v[0, :, h], g_log2[0, :, h], beta[0, :, h],
            h0[0, h], scale, BT)
        # FLA kg is [B, T, H, K]; our kd is [NT, BT, K]
        kg = fi["kg"][0, :, h].reshape(-1, BT, K)
        qg = fi["qg"][0, :, h].reshape(-1, BT, K)
        Aqk = fi["Aqk"][0, :, h].reshape(-1, BT, BT)
        Akk = fi["Akk"][0, :, h].reshape(-1, BT, BT)
        vn = fi["v_new"][0, :, h].reshape(-1, BT, V)
        hh = fi["h"][0, :, h]  # [NT, K, V]
        ba = beta[0, :, h].reshape(-1, BT)
        w = fi["w"][0, :, h].reshape(-1, BT, K)
        print(f"[head {h}]")
        for nm, a, b in [
            ("qg  vs qd/scale (=q*E)", qg, ri["qd"] / scale),
            ("kg  vs kr (k_restored)", kg, ri["kr"]),
            ("tril(Aqk) vs Mqk", torch.tril(Aqk), ri["Mqk"]),
            ("Akk vs P", Akk, ri["P"]),
            ("w   vs P*diag(beta)*kd", w, ri["P"] @ (ba[:, :, None] * ri["kd"])),
            ("v_new vs U (P*(w*beta))", vn, ri["U"]),
            ("h[0] vs S_in[0]", hh[0], ri["S_in"][0]),
            ("h[-1] vs S_in[-1]", hh[-1], ri["S_in"][-1]),
        ]:
            worst = max(worst, stats(nm, a, b)[1])
        stats("Akk vs P*diag(beta) (alt)", Akk, ri["P"] * ba[:, None, :])
    print(f"  worst max_rel (main mapping) = {worst:.3e}")
    print()


def part_b(device):
    print("=" * 78)
    print("Part B: profiled Triton kernel decomposition of FLA chunk_kda fwd+bwd")
    print("=" * 78)
    from fla.ops.kda import chunk_kda

    torch.manual_seed(0)
    B, T, H, K, V = 1, 512, 4, 128, 128
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16, requires_grad=True)
    g = -torch.rand(B, T, H, K, device=device, dtype=torch.bfloat16)
    beta = torch.rand(B, T, H, device=device, dtype=torch.bfloat16)
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32)

    def run():
        o, ht = chunk_kda(q, k, v, g, beta, initial_state=h0,
                          output_final_state=True, chunk_size=64)
        loss = o.float().pow(2).mean() + ht.float().pow(2).mean()
        loss.backward()
        for t in (q, k, v):
            t.grad = None

    run()  # warmup
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run()
    print(f"  config: B={B} T={T} H={H} K={K} V={V} chunk=64 bf16")
    evs = [e for e in prof.key_averages()
           if e.device_type == torch.autograd.DeviceType.CUDA and e.count > 0]
    evs.sort(key=lambda e: -e.self_device_time_total)
    skip = ("void at::", "Memcpy", "Memset", "at::native")
    print("  -- CUDA kernels (excluding aten elementwise/copies) --")
    for e in evs:
        if any(e.key.startswith(s) or s in e.key for s in skip):
            continue
        print(f"  {e.count:3d}x  {e.key}   ({e.self_device_time_total/1e3:.3f} ms)")
    print()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "need GPU"
    device = "cuda"
    part_a(device)
    part_b(device)
    print("DONE")
