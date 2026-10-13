"""P7-65: attribute the cost of the CUDA B2 **production kernel** (`-DB2M_PROD`, P7-60/61/63) to its
six non-GEMM segments by **cumulative ablation** inside the production kernel itself.

Motivation (docs/07 §67.4): P7-64 landed the production kernel in the full reverse chain and found
it **correct but ~1.6x slower** than the Triton fused B2/B3 (`_fused_b2b3_kernel`).  §67.4 attributed
the ~770 us overhead over the bare GEMM core to "the six non-GEMM segments accumulating in one
1-block/SM kernel", but never decomposed it -- the per-segment numbers of P7-52..P7-57 were each
measured as `core + one segment` in **separate** kernels/launches, so the production-regime
accumulation is unmeasured.  This round measures it directly.

Method: `-DB2M_PROD -DB2M_LEVEL=k` keeps every flag / launch signature / staging path intact and
only removes the segment **bodies** in reverse dependency order (add order 1=OUT 2=DKGB 3=DA
4=SAND 5=DB 6=GATE, with INTRA riding GATE).  Level 6 is the P7-60 production kernel; level 0 is the
bare GEMM core (with all staging still present).  The marginal cost `T(k)-T(k-1)` is exactly what
segment `k` adds *on top of the production kernel* -- the quantity P7-64 could not see.

  Part A  anchor: each level's surviving outputs are **bit-identical** to level 6's same outputs
          (the ablation removes only compute; every segment's output is independent of whether the
          later segments ran) -- L1 `dv2/dvb`, L2 `dkgb`, L3 `dApre`, L4 `dA`, L5 `db`, L0 `dgk`.
  Part B  timing: dev per level 0..6 on 4 shapes -> the per-segment marginal-cost curve.
  Part C  resources: ptxas regs / spill / smem per level.

Timing口径: CUPTI `dev` (chain-style, `profile_dev`), idle card.  Builds use `-DB2M_PROD -DB2M_HDW`
(the deployment form of §67).  Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_prod_levels.py \
        > .../harness/check_bwd_b2_prod_levels.out.txt 2>&1
    B2PL_MODE=a | b | c
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

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs   # noqa: E402
from check_bwd_ws_reuse import cdiv                                  # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev                        # noqa: E402

CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 16),
    ("B2 T1024 H8", 2, 1024, 8, 16),
    ("T512 H4", 1, 512, 4, 16),
    ("T250 H3 (non-div)", 1, 250, 3, 16),
]
BT = 16
SCALE = 1.0 / (128 ** 0.5)
CU = os.path.join(HARNESS, "b2_mma_probe.cu")

# Level -> removed segments (for the report).
LEVELS = list(range(7))
ADD = {1: "OUT", 2: "DKGB", 3: "DA", 4: "SAND", 5: "DB", 6: "GATE(+INTRA)"}


def level_flags(k):
    return ["-DB2M_PROD=1", "-DB2M_HDW=1", f"-DB2M_LEVEL={k}"]


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def build_v(tag, extra):
    so = f"/tmp/b2_level_{tag}.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v"] + extra + ["-lcuda", CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(f"nvcc failed for {tag}")
    return so, r.stderr


def build(tag, extra):
    return build_v(tag, extra)[0]


def load(so):
    return ctypes.CDLL(so)


def make(B, T, H, seed=1):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def run_prod(lib, inp, dw=None):
    lib.b2prod_launch.restype = ctypes.c_int
    lib.b2prod_launch.argtypes = ([ctypes.c_void_p] * 24 + [ctypes.c_float] +
                                  [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 +
                                  [ctypes.c_void_p])
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    if dw is None:
        dw = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw_out = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    dv2 = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    dvb = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    dkgb = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dqg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dkg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    db_out = torch.zeros(B, T, H, device=dev, dtype=torch.float32)
    dApre = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    dA = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    rc = lib.b2prod_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                           ptr(inp["do"]), ptr(inp["dv"]), ptr(inp["k"]), ptr(inp["g"]),
                           ptr(dw),
                           ptr(dq), ptr(dk), ptr(dw_out), ptr(dgk),
                           ptr(inp["A"]), ptr(inp["beta"]), ptr(dv2), ptr(dvb),
                           ptr(dkgb), ptr(inp["q"]), ptr(inp["k"]), ptr(inp["g"]),
                           ptr(dqg), ptr(dkg), ptr(dg), SCALE,
                           ptr(db_out), ptr(dApre), ptr(inp["A"]), ptr(dA),
                           B, T, H, NT, stream())
    if rc != 0:
        raise RuntimeError(f"b2prod_launch rc={rc}")
    return dict(dq=dq, dk=dk, dw=dw_out, dgk=dgk, dv2=dv2, dvb=dvb, dkgb=dkgb,
                dqg=dqg, dkg=dkg, dg=dg, db=db_out, dApre=dApre, dA=dA)


def bits(name, a, b, out):
    d = (a.float() - b.float()).abs().max().item()
    flag = "OK " if d == 0.0 else "DIFF"
    line = f"    {name:26s} max_abs={d:.3e}  [{flag}]"
    print(line)
    out.write(line + "\n")
    return d


# ------------------------------------------------------------------------- parts
# segment -> outputs it produces (anchor at level >= producer level, compare to level 6).
ANCHOR = {
    1: ["dv2", "dvb"],
    2: ["dkgb"],
    3: ["dApre"],
    4: ["dA"],
    5: ["db"],
}


def part_a(cases):
    print("=" * 82)
    print("Part A: each level's surviving outputs are bit-identical to level 6's (ablation = compute-only)")
    print("=" * 82)
    libs = {k: load(build(f"l{k}", level_flags(k))) for k in LEVELS}
    worst = 0.0
    with open("/tmp/b2_level_a.txt", "w") as out:
        for (name, B, T, H, seed) in cases:
            print(f"\n# {name}")
            out.write(f"\n# {name}\n")
            inp = make(B, T, H, seed)
            res = {k: run_prod(libs[k], inp) for k in (0, 1, 2, 3, 4, 5, 6)}
            full = res[6]
            for lvl, fields in ANCHOR.items():
                for f in fields:
                    worst = max(worst, bits(f"L{lvl}.{f:6s} vs L6.{f}", res[lvl][f], full[f], out))
            worst = max(worst, bits("L0.dgk     vs L6.dgk", res[0]["dgk"], full["dgk"], out))
        print(f"\n  worst: max_abs={worst:.3e}  [{'OK' if worst == 0.0 else 'MISMATCH'}]")


def ptxas_info(txt):
    reg = re.search(r"Used (\d+) registers", txt)
    spill = re.search(r"(\d+) bytes spill stores", txt)
    return (int(reg.group(1)) if reg else -1,
            int(spill.group(1)) if spill else 0)


def part_b(cases):
    print("=" * 82)
    print("Part B: dev (CUPTI) per level -> per-segment marginal cost in the production kernel")
    print("=" * 82)
    libs = {k: load(build(f"l{k}", level_flags(k))) for k in LEVELS}
    for (name, B, T, H, seed) in cases:
        inp = make(B, T, H, seed)
        dw = run_prod(libs[6], inp)["dw"]     # core dw (HDW ignores the input, but keep same operand)
        torch.cuda.synchronize()
        devs = {}
        for k in LEVELS:
            f = (lambda kk=k: run_prod(libs[kk], inp, dw))
            devs[k], _ = profile_dev(f)
        print(f"\n# {name}   (dev us per level; marginal delta = segment added over previous level)")
        print(f"  {'L':>2s} {'adds':>12s} {'dev':>9s} {'delta':>9s} {'delta%core':>11s}")
        core = devs[0]
        for k in LEVELS:
            adds = ADD.get(k, "-")
            delta = devs[k] - devs[k - 1] if k > 0 else 0.0
            pct = (delta / core * 100.0) if k > 0 else 0.0
            print(f"  {k:2d} {adds:>12s} {devs[k]:9.1f} {delta:9.1f} {pct:10.1f}%")


def part_c():
    print("=" * 82)
    print("Part C: resources (regs / spill / smem) per level")
    print("=" * 82)
    print(f"  {'L':>2s} {'adds':>12s} {'regs':>6s} {'spill':>7s} {'smem':>9s}")
    for k in LEVELS:
        so, txt = build_v(f"l{k}", level_flags(k))
        r, sp = ptxas_info(txt)
        lib = load(so)
        lib.b2mma_smem_bytes.restype = ctypes.c_int
        smem = lib.b2mma_smem_bytes()
        print(f"  {k:2d} {ADD.get(k, '-'):>12s} {r:6d} {sp:7d} {smem:9d}")


def main():
    mode = os.environ.get("B2PL_MODE", "a")
    if mode == "a":
        part_a(CASES)
    elif mode == "b":
        part_b(CASES)
    elif mode == "c":
        part_c()


if __name__ == "__main__":
    main()
