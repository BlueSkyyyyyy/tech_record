"""P7-39: land the hand-written K2-shaped CUDA B1 into the *assembled* reverse chain.

Motivation
----------
P7-15..P7-38 built, in isolation, a target-card-shaped CUDA B1 (`b1_tma_probe.cu`:
register-resident state + raw `mma.sync`/`ldmatrix` + TMA operand pipeline + TMA
`dh`/`dv2` store + `fence.proxy.async` on both reuse directions), converging to the
deployment config

    -DB1T_ETMA -DB1T_TMADH -DB1T_TMADV -DB1T_EPROXY -DB1T_SPROXY   (NSTAGE>=1)

At BT=64 it matches FLA `dhu` (bf16) to 1.15x/0.83x for the big shapes (§38).  But every
one of those rounds compared *kernel vs kernel*; the CUDA B1 was **never** plugged into
the assembled reverse chain (`dAv -> B1 -> B2(wy_dqkg) -> B3(intra) -> reverse cumsum`),
which to date has always used a *Triton* B1 (`dhu_bf16_pipe`, §25).  Two things therefore
were open:

  (i)  interface: the CUDA kernel had **no `dht` entry** (terminal state adjoint) while the
       Triton B1 does (`HAS_DHT`, P6-7).  This round adds `b1t_launch_dht` / a `dht` seed
       to the running `S`, mirroring the Triton path exactly.
  (ii) chain-level picture: does swapping B1 for the CUDA kernel keep the six gradients at
       bf16 noise, and what is the chain dev at the *deployment* granularity `BT=CHUNK=16`?

Parts
-----
  A  interface + drop-in: on the *same* workspace operands (incl. nonzero `dht`), CUDA-B1
     vs Triton-B1 (`dhu_bf16_pipe`, NS=3): `dh`/`dv2`/`dh0` max_abs (expect <= 1 bf16 ulp
     because CUDA stores `dh`/`dv2` in bf16 while the Triton chain keeps them fp32).
  B  six-gradient chain vs FLA autograd: CUDA-B1 chain (BT=16, NSTAGE=4) vs Triton-B1
     chain (BT=16, NS=3), same 6 cases as P6-8/P7-9/P7-22 (incl. non-divisible T, g=0, B=2).
  C  chain dev/wall: CUDA-B1 chain vs Triton-B1 chain vs FLA reverse, + per-kernel B1
     device attribution.
  D  isolated B1 kernel-only at the deployment granularity (BT=16): CUDA NSTAGE sweep vs
     Triton pipe NS sweep, to see the granularity-matched head-to-head.

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b1_cuda_e2e.py \
        > .../harness/check_bwd_b1_cuda_e2e.out.txt 2>&1
    B1CE_MODE=chain  # skip A/D
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_ws_reuse import stats, cdiv                                    # noqa: E402
from check_bwd_dhu_dtype import make_inputs, as_dtype, fla_dhu                # noqa: E402
from check_bwd_b1_bf16_e2e import prep, fla_golden, time_wall, profile_dev    # noqa: E402
from check_bwd_b1_pipe_e2e import (make_chain_fn_pipe, build_dhu_bf16_pipe,   # noqa: E402
                                   dhu_bf16_pipe_triton, CHUNK, D)            # noqa: E402
from check_bwd_e2e import build_dav_kernel, dav_triton                        # noqa: E402
from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton  # noqa: E402
from check_bwd_intra_kernel import build_kernel as build_intra, intra_triton  # noqa: E402
from fla.ops.utils import chunk_local_cumsum                                 # noqa: E402

CUDA_SO = "/tmp/b1_tma_deploy.so"
DEPLOY_FLAGS = ["-lcuda", "-DB1T_ETMA", "-DB1T_TMADH", "-DB1T_TMADV",
                "-DB1T_EPROXY", "-DB1T_SPROXY"]


def build():
    src = os.path.join(HARNESS, "b1_tma_probe.cu")
    so = os.environ.get("B1CE_SO", CUDA_SO)
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + DEPLOY_FLAGS + [src, "-o", so]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed")
    return so


def load(so):
    lib = ctypes.CDLL(so)
    for name in ("b1t_launch", "b1t_launch_dht"):
        f = getattr(lib, name)
        f.restype = ctypes.c_int
        # b1t_launch_dht has one extra pointer (dht) before B,T,H,V,NT
        f.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 10 + \
            [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_void_p]
    lib.b1t_launch.restype = ctypes.c_int
    lib.b1t_launch.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 9 + \
        [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_void_p]
    return lib


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def run_cuda_b1(lib, fi, do, dv, scale, BT, NSTAGE, dht=None, out_f32=True):
    """CUDA B1 drop-in.  `fi` = bf16 operands {qg,kg,w,g_cum}; `do`,`dv` = [B,T,H,V].
    Returns dh([B,NT,H,K,V]), dh0([B,H,K,V] fp32), dv2([B,T,H,V]).  The native CUDA
    kernel writes `dh`/`dv2` in bf16; `out_f32=False` keeps that (fair chain timing),
    `out_f32=True` casts back for the isolation checks."""
    qg, kg, w, g_cum = fi["qg"], fi["kg"], fi["w"], fi["g_cum"]
    B, T, HV, Kk = qg.shape
    V = do.shape[-1]
    NT = cdiv(T, BT)
    last_idx = torch.clamp((torch.arange(NT, device=qg.device) + 1) * BT - 1, max=T - 1)
    etot = torch.exp2(g_cum.float()[:, last_idx]).contiguous()
    dh = torch.empty(B, NT, HV, Kk, V, dtype=torch.bfloat16, device=qg.device)
    dh0 = torch.empty(B, HV, Kk, V, dtype=torch.float32, device=qg.device)
    dv2 = torch.empty_like(do)
    q = qg.contiguous().to(torch.bfloat16)
    k = kg.contiguous().to(torch.bfloat16)
    ww = w.contiguous().to(torch.bfloat16)
    do_ = do.contiguous().to(torch.bfloat16)
    dvv = dv.contiguous().to(torch.bfloat16)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    if dht is not None:
        dh_ = dht.contiguous().float()
        rc = lib.b1t_launch_dht(BT, NSTAGE, ptr(q), ptr(k), ptr(ww), ptr(do_), ptr(dvv),
                                ptr(etot), ptr(dh_), ptr(dh), ptr(dh0), ptr(dv2),
                                B, T, HV, V, NT, scale, stream)
    else:
        rc = lib.b1t_launch(BT, NSTAGE, ptr(q), ptr(k), ptr(ww), ptr(do_), ptr(dvv),
                            ptr(etot), ptr(dh), ptr(dh0), ptr(dv2),
                            B, T, HV, V, NT, scale, stream)
    if rc != 0:
        raise RuntimeError(f"b1t launch rc={rc} (BT={BT}, NSTAGE={NSTAGE})")
    if out_f32:
        dh = dh.float()
        dv2 = dv2.float()
    return dh, dh0, dv2


def make_chain_cuda(lib, fi, inp, qn, kn, do, dfin, scale, NSTAGE):
    dav_k = build_dav_kernel()
    dqkg_k = build_dqkg()
    intra_k = build_intra()
    v_f = inp["v"].float()
    fi_b = dict(fi)
    for kk in ("qg", "kg", "w"):
        fi_b[kk] = fi[kk].to(torch.bfloat16)
    do_b = do.to(torch.bfloat16)
    dht_c = dfin.contiguous()

    def run():
        dAqk, dv = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)
        # native bf16 `dh`/`dv2` out of the CUDA B1 (deployment dtype); B2 takes bf16 `dh`
        # directly and needs fp32 `dv` (it feeds a `tl.dot`), so only `dv2` is cast back.
        dh, dh0, dv2 = run_cuda_b1(lib, fi_b, do_b, dv, scale, CHUNK, NSTAGE,
                                   dht=dht_c, out_f32=False)
        inp2 = dict(q=qn, k=kn, v=v_f, v_new=fi["v_new"], g=fi["g_cum"], beta=fi["beta"],
                    A=fi["Akk"], h=fi["h"], do=do, dh=dh, dv=dv2.float())
        b2 = wy_dqkg_triton(dqkg_k, inp2, scale, CHUNK)
        inp3 = dict(q=qn, k=kn, g=fi["g_cum"], beta=fi["beta"],
                    dAqk=dAqk, dAkk=b2["dA"])
        b3 = intra_triton(intra_k, inp3, CHUNK, CHUNK)
        dg = chunk_local_cumsum(b2["dg"] + b3["dg"], chunk_size=CHUNK, reverse=True)
        return dict(dq=b2["dq"] + b3["dq"], dk=b2["dk"] + b3["dk"],
                    dv=b2["dv"], dg=dg, dbeta=b2["db"] + b3["db"], dh0=dh0,
                    dh=dh, dv2=dv2)

    return run


CASES = [
    ("T=128 H=2", 1, 128, 2, 0, 1.0),
    ("T=256 H=4", 1, 256, 4, 1, 1.0),
    ("T=250 H=3 (non-div)", 1, 250, 3, 2, 1.0),
    ("T=512 H=4 g=2", 1, 512, 4, 3, 2.0),
    ("B2 T=513 H=2 (non-div)", 2, 513, 2, 4, 1.0),
    ("T=192 H=2 g=0", 1, 192, 2, 5, 0.0),
]
TCASES = [
    ("T=512 H=4", 1, 512, 4, 0, 1.0),
    ("B2 T=1024 H=8", 2, 1024, 8, 1, 1.0),
    ("B8 T=2048 H=8", 8, 2048, 8, 2, 1.0),
    ("T=250 H=3 nondiv", 1, 250, 3, 3, 1.0),
]


def part_a(lib, scale, lower_bound):
    print("=" * 92)
    print("Part A: CUDA-B1 vs Triton-B1 (same workspace operands, nonzero dht=dfin)")
    print("  dh/dv2 in bf16 (CUDA) vs fp32 (Triton chain) => ~1 bf16 ulp is the floor")
    print("=" * 92)
    dhu_k = build_dhu_bf16_pipe()
    for (name, B, T, H, seed, gs) in CASES:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b = dict(fi)
        for kk in ("qg", "kg", "w"):
            fi_b[kk] = fi[kk].to(torch.bfloat16)
        do_b = do.to(torch.bfloat16)
        dAqk, dv = dav_triton(build_dav_kernel(), fi["v_new"], fi["Aqk"], do, scale, CHUNK)
        dh_t, dh0_t, dv2_t = dhu_bf16_pipe_triton(dhu_k, fi_b, do_b, dv, scale, CHUNK,
                                                   BV=32, dht=dfin, NS=3)
        dh_c, dh0_c, dv2_c = run_cuda_b1(lib, fi_b, do_b, dv, scale, CHUNK, 4, dht=dfin)
        print(f"\n# {name}")
        stats("dh  cuda vs triton", dh_c, dh_t)
        stats("dv2 cuda vs triton", dv2_c, dv2_t)
        stats("dh0 cuda vs triton", dh0_c, dh0_t)
        # dht contribution must be present: compare against dht=0 to show it matters
        dh_z, _, _ = run_cuda_b1(lib, fi_b, do_b, dv, scale, CHUNK, 4, dht=None)
        stats("dh  dht vs no-dht", dh_c, dh_z)
        del dh_t, dh0_t, dv2_t, dh_c, dh0_c, dv2_c, dh_z


def part_b(lib, scale, lower_bound):
    print("\n" + "=" * 92)
    print("Part B: six-gradient chain (CUDA-B1 NSTAGE=4 BT=16) vs Triton-B1 chain (NS=3) vs FLA")
    print("=" * 92)
    worst_c, worst_t, worst_d = {}, {}, {}
    for (name, B, T, H, seed, gs) in CASES:
        print(f"\n# {name}  B={B} T={T} H={H} seed={seed} g_scale={gs}")
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        chain_c = make_chain_cuda(lib, fi, inp, qn, kn, do, dfin, scale, NSTAGE=4)
        chain_t = make_chain_fn_pipe(fi, inp, qn, kn, do, dfin, scale, NS=3)
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        rc = chain_c()
        rt = chain_t()
        for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"),
                        ("dbeta", "beta"), ("dg", "g")]:
            _, mc = stats(f"  cuda   {fld} vs FLA", rc[fld], gold[gk])
            _, mt = stats(f"  triton {fld} vs FLA", rt[fld], gold[gk])
            _, md = stats(f"  cuda   {fld} vs triton", rc[fld], rt[fld])
            worst_c[fld] = max(worst_c.get(fld, 0.0), mc)
            worst_t[fld] = max(worst_t.get(fld, 0.0), mt)
            worst_d[fld] = max(worst_d.get(fld, 0.0), md)
        _, mc = stats("  cuda   dh0 vs FLA", rc["dh0"], gold["h0"].transpose(-1, -2))
        _, mt = stats("  triton dh0 vs FLA", rt["dh0"], gold["h0"].transpose(-1, -2))
        worst_c["dh0"] = max(worst_c.get("dh0", 0.0), mc)
        worst_t["dh0"] = max(worst_t.get("dh0", 0.0), mt)
    print("\nworst-case max_rel across cases:")
    print(f"  {'fld':6s} {'cuda vs FLA':>12s} {'triton vs FLA':>14s} {'cuda vs triton':>15s}")
    for fld in ["dq", "dk", "dv", "dbeta", "dg", "dh0"]:
        print(f"  {fld:6s} {worst_c[fld]:12.3e} {worst_t[fld]:14.3e} {worst_d.get(fld, 0.0):15.3e}")


def _b1_attrib(prof, sub):
    t, cnt = 0.0, 0
    for ev in prof.key_averages():
        if sub in ev.key:
            t += ev.self_device_time_total
            cnt += ev.count
    return t / max(cnt, 1), t, cnt


def part_c(lib, scale, lower_bound):
    from fla.ops.kda import chunk_kda
    CUDA_NS = [int(x) for x in os.environ.get("B1CE_NS", "2,4").split(",")]
    print("\n" + "=" * 100)
    print("Part C: backward-only chain timing (idle card): CUDA-B1 vs Triton-B1 vs FLA")
    print("  wall = event-pair / reps (primary); dev = CUPTI (over-attributes tiny launches).")
    print("=" * 100)
    hdr = f"  {'case':18s} | {'FLA wall':>8s} {'FLA dev':>8s} | {'Tri wall':>8s} {'Tri dev':>7s}"
    for ns in CUDA_NS:
        hdr += f" | {'cNS'+str(ns)+' wall':>10s} {'dev':>7s}"
    print(hdr)
    for (name, B, T, H, seed, gs) in TCASES:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        qa = qn.clone().requires_grad_(True)
        ka = kn.clone().requires_grad_(True)
        va = inp["v"].float().clone().requires_grad_(True)
        ba = beta_act.clone().requires_grad_(True)
        ga = g_fla.clone().requires_grad_(True)
        ha = inp["h0"].clone().requires_grad_(True)
        o, s = chunk_kda(qa, ka, va, ga, ba, scale=scale,
                         initial_state=ha.float().transpose(-1, -2), output_final_state=True,
                         use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                         use_beta_sigmoid_in_kernel=False)
        loss = (o.float() * do).sum() + (s.float() * dfin).sum()
        holders = (qa, ka, va, ba, ga, ha)

        def fla_bwd():
            loss.backward(retain_graph=True)
            for t in holders:
                t.grad = None

        chain_t = make_chain_fn_pipe(fi, inp, qn, kn, do, dfin, scale, NS=3)
        chains = {ns: make_chain_cuda(lib, fi, inp, qn, kn, do, dfin, scale, NSTAGE=ns)
                  for ns in CUDA_NS}
        chain_t()
        for c in chains.values():
            c()
        fla_bwd()
        torch.cuda.synchronize()
        w_fl = time_wall(fla_bwd); d_fl, p_fl = profile_dev(fla_bwd)
        w_t = time_wall(chain_t); d_t, p_t = profile_dev(chain_t)
        line = f"  {name:18s} | {w_fl:8.1f} {d_fl:8.1f} | {w_t:8.1f} {d_t:7.1f}"
        bl = {}
        for ns in CUDA_NS:
            w_c = time_wall(chains[ns]); d_c, p_c = profile_dev(chains[ns])
            us_b1c, tot_c, _ = _b1_attrib(p_c, "b1_tma_kernel")
            bl[ns] = (w_c, d_c, us_b1c)
            line += f" | {w_c:10.1f} {d_c:7.1f}"
        print(line)
        us_b1t, tot_t, _ = _b1_attrib(p_t, "_dhu_kernel")
        print(f"      B1 kernel dev: Triton {us_b1t:7.1f}us | " +
              " | ".join(f"CUDA NS{ns}={bl[ns][2]:6.1f}us" for ns in CUDA_NS) +
              f"  (wall Tri {w_t:.0f} vs CUDA " +
              "/".join(f"{bl[ns][0]:.0f}" for ns in CUDA_NS) + ")")


def part_d(lib, scale, lower_bound):
    print("\n" + "=" * 100)
    print("Part D: isolated B1 at deployment granularity BT=16: CUDA NSTAGE vs Triton NS")
    print("  (event-pair wall / launch, not CUPTI: tiny-kernel CUPTI attribution is bimodal")
    print("   here; wall includes ~launch overhead but is stable.  interleaved, reps=50)")
    print("=" * 100)
    dhu_k = build_dhu_bf16_pipe()
    shapes = [(1, 512, 4), (2, 1024, 8), (8, 2048, 8), (1, 250, 3)]
    ns_all = (1, 2, 3, 4)
    print(f"  {'shape':16s} | " + " ".join(f"c{ns:>3d}" for ns in ns_all) +
          " | " + " ".join(f"t{ns:>3d}" for ns in ns_all))
    for (B, T, H) in shapes:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
        fi_b = F["fi"]
        cu, tr = {}, {}
        for ns in ns_all:
            cu[ns] = time_wall(lambda ns=ns: run_cuda_b1(
                lib, fi_b, do, F["dv"], scale, 16, ns, out_f32=False), reps=50)
            tr[ns] = time_wall(lambda ns=ns: dhu_bf16_pipe_triton(
                dhu_k, fi_b, do, F["dv"], scale, 16, BV=32, NS=ns), reps=50)
        bc, bt = min(cu.values()), min(tr.values())
        print(f"  B{B} T{T} H{H:<6} | " + " ".join(f"{cu[ns]:4.0f}" for ns in ns_all) +
              " | " + " ".join(f"{tr[ns]:4.0f}" for ns in ns_all) +
              f"  best {bc:.0f}/{bt:.0f}={bc/bt:.2f}x")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    scale = 1.0 / (D ** 0.5)
    lower_bound = -5.0
    so = build()
    lib = load(so)
    mode = os.environ.get("B1CE_MODE", "")
    if mode not in ("chain", "b", "c"):
        part_a(lib, scale, lower_bound)
    if mode in ("", "b", "chain"):
        part_b(lib, scale, lower_bound)
    if mode in ("", "c", "chain"):
        part_c(lib, scale, lower_bound)
    if mode in ("", "d"):
        part_d(lib, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
