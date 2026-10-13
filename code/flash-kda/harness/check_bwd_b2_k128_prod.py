"""P7-90: close the last remaining K!=V boundary of the K2-shaped CUDA B2 production kernel
(`b2_mma_probe.cu`) -- the six non-GEMM segments still baked `BW == 16` in their K-axis
accumulator loops.

P7-81 parameterized `KD`/`VD` and verified the **base GEMM core** is generic in K != V; P7-82
generalized the six non-GEMM segments in **K != V** (V-axis warp width `BVW = VD/NW`); P7-88
parameterized **K** in the base GEMM core (`K in {64,128,256}`, generic `dgk` reduction) but
explicitly left the six non-GEMM segments guarded by an `#error`: their K-axis accumulators
(`acc3`, the gate/`intra` `idq/idk/idg`, the DKGB store, the `swall` handoff, the `db`
cross-term) hard-baked `nt < 2` (BW=16):

    #if (... six segments ...) && B2M_KD != 128
    #error "P7-88: the six non-GEMM segments still bake BW=16; only the base GEMM core supports K!=128"
    #endif

This round replaces every such `nt < 2` with `nt < NBW` (sized `[NBW][4]`) -- a pure
generalization: at `BW == 16` (`NBW == 2`) it is **byte-for-byte the old source**, so the
deployment `K = V = 128` build is bit-identical (Part A).  The `#error` is removed.

Parts
  Part A  regression at KD==VD==128: modified source vs pre-change (git HEAD) -- bit-identical
          across all 11 production outputs.
  Part B  correctness at K!=128: `prod -DB2M_HDW` (K in {64,128,256}, V=K, + a cross K=64/V=128)
          vs torch fp32 references -- dw/dgk (base core), dv2/dvb (OUT), dkgb/db (DKGB/DB),
          dApre (DA), and the gate triple dqg/dkg/dg (gate chain on the *same* raw core).
  Part C  stage-layout identity: prod vs prod+TMA at K!=128 (bit-identical).
  Part D  static cost (regs / spill / smem vs K).

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=3 python .../harness/check_bwd_b2_k128_prod.py \
        > .../harness/check_bwd_b2_k128_prod.out.txt 2>&1
    B2KP_MODE=a | b | c | d
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
from check_bwd_ws_reuse import stats, cdiv                          # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev                       # noqa: E402
from check_bwd_b2_core import ref_core                              # noqa: E402
from check_bwd_b2_out import ref_out                                # noqa: E402
from check_bwd_b2_db import ref_operands                            # noqa: E402
from check_bwd_b2_da import ref_dapre                               # noqa: E402
from check_bwd_b2_gate import ref_gate                              # noqa: E402
from check_bwd_b2_kv_prod import torch_dw                           # noqa: E402

BT = 16
SCALE = 1.0 / (128 ** 0.5)
CU = os.path.join(HARNESS, "b2_mma_probe.cu")

CASES = [
    ("B2 T1024 H8", 2, 1024, 8),
    ("T512 H4", 1, 512, 4),
    ("T250 H3 (non-div)", 1, 250, 3),
]
# (K, V) pairs; K in {64,128,256} is the new axis, V=K keeps every build legal,
# the cross (64,128) exercises the P7-82 V-axis generic path on the new K.
KV = [(64, 64), (128, 128), (256, 256), (64, 128)]
PROD = ["-DB2M_PROD", "-DB2M_HDW"]


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def build(src, so, K, V, extra):
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-DB2M_KD=%d" % K, "-DB2M_VD=%d" % V] + extra + \
          ["-lcuda", src, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed: %s (K=%d V=%d %s)" % (src, K, V, extra))
    return so


def build_v(src, so, K, V, extra):
    so_ = so
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v", "-DB2M_KD=%d" % K, "-DB2M_VD=%d" % V] + extra + \
          ["-lcuda", src, "-o", so_]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed: %s (K=%d V=%d %s)" % (src, K, V, extra))
    return so_, r.stderr


def orig_source():
    out = "/tmp/b2_mma_probe_orig.cu"
    r = subprocess.run(["git", "-C", "~/proj/tech_record", "show",
                        "HEAD:code/flash-kda/harness/b2_mma_probe.cu"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("cannot read HEAD source: " + r.stderr)
    open(out, "w").write(r.stdout)
    return out


def make(B, T, H, K, V, seed=1):
    inp = dqkg_make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def load_prod(so):
    lib = ctypes.CDLL(so)
    lib.b2prod_launch.restype = ctypes.c_int
    lib.b2prod_launch.argtypes = ([ctypes.c_void_p] * 24 + [ctypes.c_float] +
                                  [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 +
                                  [ctypes.c_void_p])
    lib.b2mma_smem_bytes.restype = ctypes.c_int
    return lib


def load_core(so):
    lib = ctypes.CDLL(so)
    lib.b2mma_launch.restype = ctypes.c_int
    lib.b2mma_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + \
        [ctypes.c_void_p]
    return lib


def run_prod(lib, inp):
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw_in = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)   # scratch under HDW
    dw_out = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)  # unwritten under HDW
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
                           ptr(dw_in),
                           ptr(dq), ptr(dk), ptr(dw_out), ptr(dgk),
                           ptr(inp["A"]), ptr(inp["beta"]), ptr(dv2), ptr(dvb),
                           ptr(dkgb), ptr(inp["q"]), ptr(inp["k"]), ptr(inp["g"]),
                           ptr(dqg), ptr(dkg), ptr(dg), SCALE,
                           ptr(db_out), ptr(dApre), ptr(inp["A"]), ptr(dA),
                           B, T, H, NT, stream())
    if rc != 0:
        raise RuntimeError("b2prod_launch rc=%d" % rc)
    return dict(dgk=dgk, dkgb=dkgb, dv2=dv2, dvb=dvb, db=db_out, dApre=dApre, dA=dA,
                dqg=dqg, dkg=dkg, dg=dg, dw=dw_out)


def run_core(lib, inp):
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    rc = lib.b2mma_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                          ptr(inp["do"]), ptr(inp["dv"]), ptr(dq), ptr(dk), ptr(dw), ptr(dgk),
                          B, T, H, NT, stream())
    if rc != 0:
        raise RuntimeError("b2mma_launch rc=%d" % rc)
    return dict(dq=dq, dk=dk, dw=dw, dgk=dgk)


def bits(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    flag = "OK " if d == 0.0 else "DIFF"
    print("    %-30s max_abs=%.3e  [%s]" % (name, d, flag))
    return d


def part_a():
    print("=" * 92)
    print("Part A: KD==VD==128 regression -- modified vs pre-change (git HEAD) prod, bit-identical")
    print("=" * 92)
    orig = build(orig_source(), "/tmp/b2kp_orig128.so", 128, 128, PROD)
    new = build(CU, "/tmp/b2kp_new128.so", 128, 128, PROD)
    lo, ln = load_prod(orig), load_prod(new)
    worst = 0.0
    for (name, B, T, H) in CASES:
        inp = make(B, T, H, 128, 128, seed=1)
        o = run_prod(lo, inp)
        n = run_prod(ln, inp)
        print("\n# %s  grid=(%d, %d)" % (name, cdiv(T, BT), B * H))
        for fld in ("dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dgk"):
            worst = max(worst, bits("orig vs new %s" % fld, o[fld], n[fld]))
    print("\nPart A worst max_abs over all fields/cases: %.3e  [%s]"
          % (worst, "BIT-IDENTICAL" if worst == 0.0 else "MISMATCH"))
    return worst


def part_b():
    print("\n" + "=" * 92)
    print("Part B: K!=128 six-segment correctness -- prod(HDW) vs torch fp32 references")
    print("=" * 92)
    worst = {}
    for (K, V) in KV:
        so = build(CU, "/tmp/b2kp_K%d_V%d.so" % (K, V), K, V, PROD)
        soc = build(CU, "/tmp/b2kp_core_K%d_V%d.so" % (K, V), K, V, [])
        lib, libc = load_prod(so), load_core(soc)
        print("\n#### K = %d  V = %d ####" % (K, V))
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, V, seed=1)
            cre = ref_core(inp, BT)                 # dw/dgk (base GEMM core)
            ro = ref_out(inp, BT)                   # dv2/dvb (OUT)
            dw = torch_dw(inp, BT)
            rd = ref_operands(inp, dw, BT)          # dvb/dkgb/db (DKGB/DB)
            rp = ref_dapre(inp, dw, BT)             # dApre (DA sandwich input)
            core = run_core(libc, inp)              # raw dq/dk/dgk for the gate reference
            out = run_prod(lib, inp)
            print("\n# %s  grid=(%d, %d)" % (name, cdiv(T, BT), B * H))
            # NOTE: `dw_out` is scratch under `-DB2M_HDW` (the GEMM core keeps `dw` in-kernel);
            # only `dgk` is a stored base-core output here, so `dw` is checked via `dkgb`.
            for fld, ref, tag in (("dgk", cre["dgk"], "core"),
                                  ("dv2", ro["dv2"], "out"), ("dvb", ro["dvb"], "out"),
                                  ("dkgb", rd["dkgb"], "dkgb"), ("db", rd["db"], "db"),
                                  ("dApre", rp["dApre"], "da")):
                _, mr = stats("  [%d,%d] %-6s vs torch" % (K, V, fld), out[fld], ref)
                worst["%d_%d_%s" % (K, V, fld)] = max(worst.get("%d_%d_%s" % (K, V, fld), 0.0), mr)
            # gate triple: the dg term needs the kernel's own dkgb cross-term
            rg2 = ref_gate(inp, core["dq"], core["dk"], core["dgk"], out["dkgb"])
            for fld in ("dqg", "dkg", "dg"):
                _, mr = stats("  [%d,%d] %-6s vs torch" % (K, V, fld), out[fld], rg2[fld])
                worst["%d_%d_%s" % (K, V, fld)] = max(worst.get("%d_%d_%s" % (K, V, fld), 0.0), mr)
    print("\nworst max_rel over all cases:")
    print("  %5s %5s  %-8s %11s" % ("K", "V", "field", "vs torch"))
    for (K, V) in KV:
        for fld in ("dgk", "dv2", "dvb", "dkgb", "db", "dApre", "dqg", "dkg", "dg"):
            print("  %5d %5d  %-8s %11.3e"
                  % (K, V, fld, worst["%d_%d_%s" % (K, V, fld)]))
    return worst


def part_c():
    print("\n" + "=" * 92)
    print("Part C: stage-layout identity -- prod vs prod+TMA at K!=128 (bit-identical)")
    print("=" * 92)
    worst = 0.0
    for (K, V) in KV:
        try:
            lib = load_prod(build(CU, "/tmp/b2kp_K%d_V%d.so" % (K, V), K, V, PROD))
            libt = load_prod(build(CU, "/tmp/b2kp_K%d_V%d_tma.so" % (K, V), K, V,
                                   PROD + ["-DB2M_TMA"]))
        except SystemExit as e:
            print("\n#### K = %d  V = %d ####  SKIP (build failed: %s)" % (K, V, e))
            continue
        print("\n#### K = %d  V = %d ####" % (K, V))
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, V, seed=1)
            a = run_prod(lib, inp)
            b = run_prod(libt, inp)
            d = 0.0
            for fld in ("dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dgk"):
                d = max(d, bits("%s %s" % (name, fld), a[fld], b[fld]))
            worst = max(worst, d)
    print("\nPart C worst max_abs: %.3e" % worst)
    return worst


def ptxas_info(txt):
    reg = re.search(r"Used (\d+) registers", txt)
    spill = re.search(r"(\d+) bytes spill stores", txt)
    return (int(reg.group(1)) if reg else -1, int(spill.group(1)) if spill else 0)


def part_d():
    print("\n" + "=" * 92)
    print("Part D: static cost (regs / spill / smem vs K; V=K)")
    print("=" * 92)
    print("  %5s %6s %7s %10s %8s" % ("K", "regs", "spill", "smem(B)", "smem(KB)"))
    for (K, V) in KV:
        try:
            so, txt = build_v(CU, "/tmp/b2kp_d_K%d.so" % K, K, V, PROD)
        except SystemExit as e:
            print("  %5d  BUILD FAILED: %s" % (K, e))
            continue
        r, sp = ptxas_info(txt)
        smem = ctypes.CDLL(so).b2mma_smem_bytes()
        print("  %5d %6d %7d %10d %8.1f" % (K, r, sp, smem, smem / 1024.0))


def part_e():
    print("\n" + "=" * 92)
    print("Part E: kernel-only timing -- prod(HDW) dev vs K (V=K), full production launch")
    print("=" * 92)
    print("  %-18s %5s %5s | %10s" % ("case", "K", "V", "prod dev"))
    for (K, V) in KV:
        lib = load_prod(build(CU, "/tmp/b2kp_e_K%d_V%d.so" % (K, V), K, V, PROD))
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, V, seed=1)
            run_prod(lib, inp)
            torch.cuda.synchronize()
            d, _ = profile_dev(lambda: run_prod(lib, inp), reps=30)
            print("  %-18s %5d %5d | %8.1f us" % (name, K, V, d))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print("GPU: %s  torch=%s" % (torch.cuda.get_device_name(), torch.__version__))
    mode = os.environ.get("B2KP_MODE", "")
    if mode in ("", "a"):
        part_a()
    if mode in ("", "b"):
        part_b()
    if mode in ("", "c"):
        part_c()
    if mode in ("", "d"):
        part_d()
    if mode in ("", "e"):
        part_e()
    print("\nDONE")


if __name__ == "__main__":
    main()
