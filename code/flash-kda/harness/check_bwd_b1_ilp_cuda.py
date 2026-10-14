"""P7-18: isolate the **mma-dependency / parallelism** term of the hand-written CUDA B1.

§20 (P7-17) closed the bank-conflict hypothesis: forcing the shared-load conflicts from
76% down to ~19% (smem padding) bought only 10-23% end to end and left the `wait` stall
(1.79 -> 1.77) and the `Eligible Warps/Scheduler` (0.15-0.26) untouched.  §20.6 therefore
demoted bank conflict to a ~1.2x sub-term and named the *magnitude* gap (xFLA 4.9-8x) as
**parallelism / mma density** (tensor 2.3% vs FLA 14%), with the concrete first priority:

    "把「每 CTA 2 warp、每 warp 一条 mma 链」做到 K2 形状（多 warp / warp-specialized、
     每 chunk 更高的 mma 密度）——这是量级项。"

This script is the cheapest possible controlled test of that claim **before** committing to
a K2-shaped rewrite.  Two orthogonal, free-ish knobs on the same source:

  A) mma instruction-level parallelism *within* a warp -- `-DB1_ILP1` (op1: kk-outer/m-inner,
     exposing MT1 independent `k @ S` accumulator chains) and `-DB1_ILP2` (op2/op3: kk-outer,
     all 2*MT2 cq/cw chains live).  Loop order only: each accumulator still sums kk in the
     same order, so it must be **bit-identical**; costs zero smem, only registers.
  B) parallelism *across* warps/CTAs -- `-DB1_BV/-DB1_NW` (BW=BV/NW fixed at 16): BV=16/NW=1
     gives 8x more CTAs (better SM fill for small B*H), BV=64/NW=4 fewer but fatter CTAs.

  Part A  correctness: every variant must be bit-identical to the baseline (and to FLA)
  Part B  kernel-only CUPTI us/call: ILP axis (base/ilp1/ilp2/ilp12) x BT, and BV axis
  Part C  smem footprint per variant
  Part D  register/occupancy (`cuobjdump -res-usage`) -- ILP2 needs 2*MT2 accumulator frags

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b1_ilp_cuda.py
ncu pairing of base vs ilp1/ilp2:
    bash .../harness/probe_b1_cuda_ilp_ncu.sh > .../probe_b1_cuda_ilp_ncu.out.txt 2>&1
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

from check_bwd_dhu_dtype import (                                          # noqa: E402
    cuda_kernel_us, stats, make_inputs, as_dtype, fla_dhu,
)
from check_bwd_b1_cuda import run_cuda                                    # noqa: E402
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu    # noqa: E402

D = 128
SRC_CU = os.path.join(HARNESS, "b1_cuda_probe.cu")

# name -> extra nvcc flags
VARIANTS = {
    "base":  [],
    "ilp1":  ["-DB1_ILP1"],
    "ilp2":  ["-DB1_ILP2"],
    "ilp12": ["-DB1_ILP1", "-DB1_ILP2"],
    "bv16":  ["-DB1_BV=16", "-DB1_NW=1"],
    "bv64":  ["-DB1_BV=64", "-DB1_NW=4"],
}


def build(tag, extra=()):
    so = f"/tmp/b1_cuda_{tag}.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + list(extra) + [SRC_CU, "-o", so]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"nvcc failed for {tag}")
    lib = ctypes.CDLL(so)
    lib.b1_launch.restype = ctypes.c_int
    lib.b1_launch.argtypes = [ctypes.c_int, ctypes.c_int] + \
        [ctypes.c_void_p] * 9 + [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_void_p]
    lib.b1_smem_bytes.restype = ctypes.c_int
    lib.b1_smem_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
    return lib, so


def res_usage(so):
    """Parse `cuobjdump -res-usage` (REG/STACK/LOCAL) per template instantiation."""
    import re
    r = subprocess.run(["cuobjdump", "-res-usage", so], capture_output=True, text=True)
    if r.returncode != 0:
        return {}
    out, cur = {}, None
    for line in r.stdout.splitlines():
        if line.strip().startswith("Function "):
            cur = line.strip()[len("Function "):].rstrip(":")
        m = re.search(r"REG:(\d+)", line)
        if m and cur:
            st = re.search(r"STACK:(\d+)", line)
            lo = re.search(r"LOCAL:(\d+)", line)
            out[cur] = (int(m.group(1)), int(st.group(1)) if st else -1,
                        int(lo.group(1)) if lo else -1)
            cur = None
    return out


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs, sos = {}, {}
    for tag, extra in VARIANTS.items():
        libs[tag], sos[tag] = build(tag, extra)

    scale = D ** -0.5

    # ---------------------------------------------------------------- Part C
    print("\n" + "=" * 84)
    print("Part C: smem footprint (bytes, BT=64/NS=1) per variant")
    print("=" * 84)
    for tag in VARIANTS:
        print(f"  {tag:6s} smem(BT64,NS1) = {libs[tag].b1_smem_bytes(64, 1):8d}   "
              f"(BT32,NS1) = {libs[tag].b1_smem_bytes(32, 1):8d}")

    # ---------------------------------------------------------------- Part D
    print("\n" + "=" * 84)
    print("Part D: register spill (cuobjdump -res-usage), BT=64 instantiation")
    print("=" * 84)
    for tag in VARIANTS:
        ru = res_usage(sos[tag])
        rows = {k: v for k, v in ru.items() if "ILi64ELi1E" in k or "ILi32ELi1E" in k}
        if not rows:
            print(f"  {tag:6s} (no res-usage parsed)")
            continue
        for k, v in sorted(rows.items()):
            inst = "BT64/NS1" if "ILi64ELi1E" in k else "BT32/NS1"
            print(f"  {tag:6s} {inst:9s} REG={v[0]:3d} STACK={v[1]} LOCAL={v[2]}")

    # ---------------------------------------------------------------- Part A
    print("\n" + "=" * 84)
    print("Part A: ILP loop reorder must be numerically inert (bit-identical to base / FLA)")
    print("=" * 84)
    for (B, T, H) in [(1, 512, 4), (2, 1024, 8), (1, 250, 3)]:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        for BT in (32, 64):
            F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
            outs = {}
            for tag, lib in libs.items():
                outs[tag] = run_cuda(lib, F["fi"], do, F["dv"], scale, BT, 1)
            print(f"  B={B} T={T} H={H} BT={BT}")
            for tag in ("ilp1", "ilp2", "ilp12", "bv16", "bv64"):
                for i, nm in enumerate(("dh", "dh0", "dv2")):
                    r = (outs[tag][i].float() - outs["base"][i].float()).abs().max().item()
                    if r != 0.0:
                        print(f"      {tag:6s} {nm} vs base max_abs={r:g}  <-- NOT bit-identical")
            print("      ilp1/ilp2/ilp12/bv16/bv64 vs base: 逐位相同（无输出=全 0）"
                  if all((outs[tag][i].float() - outs["base"][i].float()).abs().max().item() == 0.0
                         for tag in ("ilp1", "ilp2", "ilp12", "bv16", "bv64") for i in range(3))
                  else "      <-- 见上")

    # ---------------------------------------------------------------- Part B
    print("\n" + "=" * 84)
    print("Part B: kernel-only CUPTI us/call — ILP axis (BT=64) and BV axis vs FLA dhu")
    print("=" * 84)
    shapes = [(1, 512, 4), (2, 1024, 8), (8, 2048, 8)]
    for (B, T, H) in shapes:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        for BT in (32, 64):
            F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
            fla_us = cuda_kernel_us(lambda F=F, BT=BT: chunk_gated_delta_rule_bwd_dhu(
                q=F["fi"]["qg"], k=F["fi"]["kg"], w=F["fi"]["w"], gk=F["fi"]["g_cum"],
                h0=h0, dht=None, do=do, dv=F["dv"], scale=scale, chunk_size=BT),
                "bwd_kernel_dhu")
            print(f"  B{B} T{T} H{H}  BT={BT}   FLA dhu = {fla_us:7.1f}us")
            base = None
            for tag in ("base", "ilp1", "ilp2", "ilp12", "bv16", "bv64"):
                try:
                    us = cuda_kernel_us(lambda F=F, lib=libs[tag], BT=BT: run_cuda(
                        lib, F["fi"], do, F["dv"], scale, BT, 1), "b1_dhu_cuda")
                except RuntimeError as e:
                    print(f"      {tag:6s} failed ({e})")
                    continue
                if tag == "base":
                    base = us
                xtag = f"xFLA={us/fla_us:5.2f}"
                xb = f"  xbase={us/base:5.2f}" if base else ""
                print(f"      {tag:6s} {us:8.1f}us   {xtag}{xb}")
    print("\nDONE")


if __name__ == "__main__":
    main()
