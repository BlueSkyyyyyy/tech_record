"""P7-79 (chain): land the **native bf16 mma** (`-DB2M_BF16MMA`) -- and the **mma-ised DA**
(`-DB2M_DAMMA`) -- into the bf16 **folded production** reverse chain.

Context
-------
P7-75 (`check_bwd_b2_bf16_e2e.py`) landed `-DB2M_BF16` into the chain (`pcb`/`pcib`), but left
two boundaries (docs/07 §78.5 / §81.6): the chain variants were **非 DAMMA** and the *native*
bf16 mma (`-DB2M_BF16MMA`, P7-74) was only measured kernel-only (`check_bwd_b2_bf16_prod.py`
Part C: `prod_bf16_mma` 0.659x vs `prod`), never chained.  §81.4 explicitly recommends
"...若追求 mma 密度可叠 `-DB2M_BF16MMA`".  This round closes that: chain

    dAv(Triton) -> CUDA-B1 -> CUDA-B2-prod-fold[-DB2M_BF16][-DB2M_BF16MMA][-DB2M_DAMMA] -> reverse cumsum

only eating FlashKDA's 16-chunk K1 workspace, and check six gradients vs `torch.autograd`(FLA)
and vs the `pcib` (tf32-mma bf16) baseline.

Variants
--------
  `pc`     : fp32 `-DB2M_PROD -DB2M_HDW` + standalone Triton B3 intra  (P7-64 baseline)
  `pci`    : fp32 `-DB2M_PROD -DB2M_HDW -DB2M_INTRA` (1 launch)
  `pcb`    : bf16 prod (non-fold) + Triton B3 intra
  `pcib`   : bf16 fold (`-DB2M_BF16 -DB2M_INTRA`)            -- P7-75 baseline
  `pcibm`  : `+ -DB2M_BF16MMA`  (native bf16 `mma.sync.m16n8k16`)
  `pcibmd` : `+ -DB2M_BF16MMA -DB2M_DAMMA` (mma-ised DA too)
  `cf`     : the §45 best chain (CUDA-B1 + fused Triton B2/B3)

Parts
-----
  A  correctness: six grads vs FLA autograd (worst max_rel), plus the variant-vs-variant deltas
     (`pcibm`/`pcibmd` vs `pcib`, expect only fp32 accumulation-order / tf32-DA differences), 6
     cases incl. non-divisible T, g=0, B=2.
  B  chain dev/wall (idle card): FLA vs cf vs pc/pci/pcb/pcib/pcibm/pcibmd, per-kernel attribution.
  C  static resources (ptxas): regs / spill / smem of the six CUDA B2 builds.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_bf16mma_e2e.py \
        > .../harness/check_bwd_b2_bf16mma_e2e.out.txt 2>&1
    BM_MODE=correctness | timing | res
"""
import ctypes
import os
import re
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_ws_reuse import stats, CHUNK, D                                    # noqa: E402
from check_bwd_b1_bf16_e2e import prep, fla_golden, time_wall, profile_dev        # noqa: E402
from check_bwd_b1_cuda_e2e import (build as build_b1, load as load_so,            # noqa: E402
                                   run_cuda_b1, CASES, TCASES, DEPLOY_FLAGS)
from check_bwd_prod_e2e import make_chain_prod, FOLD_FLAGS, PROD_FLAGS            # noqa: E402
from check_bwd_b2_bf16_e2e import make_chain_bf16, BF16_KEYS                      # noqa: E402
from check_bwd_cudab1_fused_e2e import make_chain_cuda_fused                      # noqa: E402

CU = os.path.join(HARNESS, "b2_mma_probe.cu")

# build flags
FP_PROD = PROD_FLAGS
FP_FOLD = FOLD_FLAGS
B_PROD = PROD_FLAGS + BF16_KEYS                       # bf16, non-fold
B_FOLD = FOLD_FLAGS + BF16_KEYS                       # bf16, fold  (P7-75 `pcib`)
B_FOLD_MMA = B_FOLD + ["-DB2M_BF16MMA=1"]             # + native bf16 mma
B_FOLD_MMA_DA = B_FOLD_MMA + ["-DB2M_DAMMA=1"]        # + mma-ised DA

BUILDS = {
    "fp_prod": (FP_PROD, False),
    "fp_fold": (FP_FOLD, True),
    "b_prod": (B_PROD, False),
    "b_fold": (B_FOLD, True),
    "b_fold_mma": (B_FOLD_MMA, True),
    "b_fold_mma_da": (B_FOLD_MMA_DA, True),
}


def build_b2(tag, extra):
    so = f"/tmp/b2_bf16mma_{tag}.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v"] + extra + ["-lcuda", CU, "-o", so]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"nvcc failed for {tag}")
    reg = re.search(r"Used (\d+) registers", r.stderr)
    spill = re.search(r"(\d+) bytes spill stores", r.stderr)
    return so, dict(reg=int(reg.group(1)) if reg else -1,
                    spill=int(spill.group(1)) if spill else 0)


def build_all():
    libs, res = {}, {}
    for tag, (extra, _fold) in BUILDS.items():
        so, info = build_b2(tag, extra)
        libs[tag] = ctypes.CDLL(so)
        res[tag] = info
    return libs, res


def _chain(lib_b1, lib, tag, fi, inp, qn, kn, do, dfin, scale):
    fold = BUILDS[tag][1]
    if tag in ("fp_prod", "fp_fold"):
        return make_chain_prod(lib_b1, lib, fold, fi, inp, qn, kn, do, dfin, scale, 4)
    return make_chain_bf16(lib_b1, lib, fold, fi, inp, qn, kn, do, dfin, scale, 4)


def part_a(lib_b1, libs, scale, lower_bound):
    print("=" * 100)
    print("Part A: bf16-native-mma folded chains vs FLA autograd, and vs the tf32-mma `pcib`")
    print("=" * 100)
    variants = ["pcb", "pcib", "pcibm", "pcibmd"]
    tag_of = {"pcb": "b_prod", "pcib": "b_fold", "pcibm": "b_fold_mma", "pcibmd": "b_fold_mma_da"}
    worst = {k: {} for k in variants + ["pcibm_vs_pcib", "pcibmd_vs_pcib"]}
    cases = CASES[:int(os.environ.get("BM_CASES", len(CASES)))]
    for (name, B, T, H, seed, gs) in cases:
        print(f"\n# {name}  B={B} T={T} H={H} seed={seed} g_scale={gs}")
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        res = {v: _chain(lib_b1, libs[tag_of[v]], tag_of[v], fi, inp, qn, kn, do, dfin, scale)()
               for v in variants}
        for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"),
                        ("dbeta", "beta"), ("dg", "g")]:
            for v in variants:
                _, m = stats(f"  {v:7s} {fld} vs FLA", res[v][fld], gold[gk])
                worst[v][fld] = max(worst[v].get(fld, 0.0), m)
            _, d1 = stats(f"  pcibm  {fld} vs pcib", res["pcibm"][fld], res["pcib"][fld])
            _, d2 = stats(f"  pcibmd {fld} vs pcib", res["pcibmd"][fld], res["pcib"][fld])
            worst["pcibm_vs_pcib"][fld] = max(worst["pcibm_vs_pcib"].get(fld, 0.0), d1)
            worst["pcibmd_vs_pcib"][fld] = max(worst["pcibmd_vs_pcib"].get(fld, 0.0), d2)
        for v in variants:
            _, m = stats(f"  {v:7s} dh0 vs FLA", res[v]["dh0"], gold["h0"].transpose(-1, -2))
            worst[v]["dh0"] = max(worst[v].get("dh0", 0.0), m)
        del res
    print("\nworst-case max_rel across cases:")
    print(f"  {'fld':6s} " + " ".join(f"{v:>11s}" for v in variants) +
          f" {'mma_vs_pcib':>12s} {'mda_vs_pcib':>12s}")
    for fld in ["dq", "dk", "dv", "dbeta", "dg", "dh0"]:
        row = " ".join(f"{worst[v].get(fld, 0.0):11.3e}" for v in variants)
        print(f"  {fld:6s} {row} {worst['pcibm_vs_pcib'].get(fld, 0.0):12.3e}"
              f" {worst['pcibmd_vs_pcib'].get(fld, 0.0):12.3e}")


def _attrib(prof, sub):
    t, cnt = 0.0, 0
    for ev in prof.key_averages():
        if sub in ev.key:
            t += ev.self_device_time_total
            cnt += ev.count
    return t / max(cnt, 1), t, cnt


def part_b(lib_b1, libs, scale, lower_bound):
    from fla.ops.kda import chunk_kda
    print("\n" + "=" * 112)
    print("Part B: backward-only chain dev/wall (idle card): FLA vs cf vs pc/pci/pcb/pcib/pcibm/pcibmd")
    print("  wall = event-pair / reps (primary); dev = CUPTI over-attributes tiny launches.")
    print("=" * 112)
    variants = ["pc", "pci", "pcb", "pcib", "pcibm", "pcibmd"]
    tag_of = {"pc": "fp_prod", "pci": "fp_fold", "pcb": "b_prod", "pcib": "b_fold",
              "pcibm": "b_fold_mma", "pcibmd": "b_fold_mma_da"}
    print(f"  {'case':18s} | {'FLA':>7s} {'cf':>7s} | " +
          " ".join(f"{v:>7s}" for v in variants) +
          f" | {'pcibm/pcib':>10s} {'pcibm/cf':>9s}")
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

        ch = {v: _chain(lib_b1, libs[tag_of[v]], tag_of[v], fi, inp, qn, kn, do, dfin, scale)
              for v in variants}
        ch_cf = make_chain_cuda_fused(lib_b1, fi, inp, qn, kn, do, dfin, scale, NSTAGE=4)
        for f in ch.values():
            f()
        ch_cf(); fla_bwd()
        torch.cuda.synchronize()
        d_fl, _ = profile_dev(fla_bwd)
        d_cf, p_cf = profile_dev(ch_cf)
        dev = {v: profile_dev(ch[v]) for v in variants}
        wall = {v: time_wall(ch[v]) for v in variants}
        row = " ".join(f"{dev[v][0]:7.1f}" for v in variants)
        print(f"  {name:18s} | {d_fl:7.1f} {d_cf:7.1f} | {row} |"
              f" {dev['pcibm'][0]/dev['pcib'][0]:10.3f} {dev['pcibm'][0]/d_cf:9.3f}")
        print(f"      wall pcib/pcibm/pcibmd = {wall['pcib']:.1f}/{wall['pcibm']:.1f}/"
              f"{wall['pcibmd']:.1f} us | cf {time_wall(ch_cf):.1f} | "
              f"pcibm kernels: B1 {_attrib(dev['pcibm'][1], 'b1_tma_kernel')[0]:.1f}"
              f" prod {_attrib(dev['pcibm'][1], 'b2mma_kernel')[0]:.1f}")


def part_c(res):
    print("\n" + "=" * 82)
    print("Part C: static resources (ptxas) of the six CUDA B2 builds")
    print("  (dynamic smem / achieved occupancy are reported by Part D `ncu`; this ptxas build")
    print("   only prints regs+spill, so blocks/SM here is the *register* limit, 256 threads)")
    print("=" * 82)
    print(f"  {'build':16s} {'regs':>6s} {'spill':>7s}  reg-limit blocks/SM (65536 / regs*256)")
    for tag in BUILDS:
        r = res[tag]
        blk_reg = 65536 // (r["reg"] * 256) if r["reg"] > 0 else 0
        print(f"  {tag:16s} {r['reg']:6d} {r['spill']:7d}  {blk_reg}")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    print("B1 flags:", " ".join(DEPLOY_FLAGS))
    scale = 1.0 / (D ** 0.5)
    lower_bound = -5.0
    print("building CUDA B1 ...")
    lib_b1 = load_so(build_b1())
    print("building CUDA B2 variants ...")
    libs, res = build_all()
    mode = os.environ.get("BM_MODE", "")
    if mode in ("", "res"):
        part_c(res)
    if mode in ("", "correctness"):
        part_a(lib_b1, libs, scale, lower_bound)
    if mode in ("", "timing"):
        part_b(lib_b1, libs, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
