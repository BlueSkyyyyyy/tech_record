"""P7-19: register-resident, `mma.sync`+`ldmatrix` hand-written CUDA B1 (`dhu`) prototype.

§21.6 concluded the *magnitude* residual to FLA `dhu` (4.9-8x) is "每 warp 的有效 mma
密度" (tensor 2.3% vs FLA 14%) and that neither bank conflict (§20, ~1.2x) nor warp-level
ILP (§21, ~1.1x) nor CTA shape closes it.  This prototype acts on that conclusion directly:
it carries the running state S **in registers** as mma accumulator fragments and issues raw
`mma.sync.m16n8k16` fed by `ldmatrix` from XOR-padded (conflict-free) smem, eliminating the
fp32 smem master and the fp32 staging round-trips of `b1_cuda_probe.cu`.

  Part A  correctness vs FLA `dhu` (bf16, same BT), incl. non-divisible T
  Part B  kernel-only CUPTI timing, NSTAGE = 1/2/3, BT = 16/32/64, vs FLA `dhu` and vs
          the *base* wmma prototype (`b1_cuda_probe.cu`) on the same operands
  Part C  smem footprint vs base

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b1_reg.py
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dhu_dtype import (                                          # noqa: E402
    cuda_kernel_us, stats, make_inputs, as_dtype, fla_dhu, cdiv,
)

D = 128
REG_SO = "/tmp/b1_reg.so"
BASE_SO = "/tmp/b1_cuda.so"


def build(src_name, so, extra=()):
    src = os.path.join(HARNESS, src_name)
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + list(extra) + [src, "-o", so]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"nvcc failed: {src_name}")


def load(so, launcher, smem_fn):
    lib = ctypes.CDLL(so)
    f = getattr(lib, launcher)
    f.restype = ctypes.c_int
    f.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 9 + \
        [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_void_p]
    s = getattr(lib, smem_fn)
    s.restype = ctypes.c_int
    s.argtypes = [ctypes.c_int, ctypes.c_int]
    return lib


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def run(lib, fi, do, dv, scale, BT, NSTAGE, launcher="b1r_launch"):
    qg, kg, w, g_cum = fi["qg"], fi["kg"], fi["w"], fi["g_cum"]
    B, T, HV, Kk = qg.shape
    V = do.shape[-1]
    NT = cdiv(T, BT)
    last_idx = torch.clamp((torch.arange(NT, device=qg.device) + 1) * BT - 1, max=T - 1)
    etot = torch.exp2(g_cum.float()[:, last_idx]).contiguous()
    dh = torch.empty(B, NT, HV, Kk, V, dtype=torch.bfloat16, device=qg.device)
    dh0 = torch.empty(B, HV, Kk, V, dtype=torch.float32, device=qg.device)
    dv2 = torch.empty_like(do)
    q = qg.contiguous(); k = kg.contiguous(); ww = w.contiguous()
    do_ = do.contiguous(); dvv = dv.contiguous()
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = getattr(lib, launcher)(BT, NSTAGE, ptr(q), ptr(k), ptr(ww), ptr(do_), ptr(dvv),
                                ptr(etot), ptr(dh), ptr(dh0), ptr(dv2),
                                B, T, HV, V, NT, scale, stream)
    if rc != 0:
        raise RuntimeError(f"{launcher} rc={rc} (BT={BT}, NSTAGE={NSTAGE})")
    return dh, dh0, dv2


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    build("b1_reg_probe.cu", REG_SO)
    build("b1_cuda_probe.cu", BASE_SO)
    lib_reg = load(REG_SO, "b1r_launch", "b1r_smem_bytes")
    lib_base = load(BASE_SO, "b1_launch", "b1_smem_bytes")
    scale = D ** -0.5
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu

    # ------------------------------------------------------------------ Part C
    print("\n" + "=" * 88)
    print("Part C: smem footprint (bytes) per (BT, NSTAGE)  [reg vs base]")
    print("=" * 88)
    for BT in (16, 32, 64):
        row = []
        for ns in (1, 2, 3):
            row.append(f"ns={ns}: {lib_reg.b1r_smem_bytes(BT, ns):6d}/{lib_base.b1_smem_bytes(BT, ns):6d}")
        print(f"  BT={BT:3d}   " + "   ".join(row))

    # ------------------------------------------------------------------ Part A
    print("\n" + "=" * 88)
    print("Part A: register-resident B1 vs FLA dhu (bf16, same BT) — correctness")
    print("=" * 88)
    for (B, T, H) in [(1, 512, 4), (2, 1024, 8), (1, 250, 3)]:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        for BT in (32, 64):
            F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
            dh, dh0, dv2 = run(lib_reg, F["fi"], do, F["dv"], scale, BT, 1)
            print(f"  B={B} T={T} H={H} BT={BT}")
            stats("dh  reg vs FLA", dh, F["dh"])
            stats("dv2 reg vs FLA", dv2, F["dv2"])
            stats("dh0 reg vs FLA", dh0, F["dh0"])

    # ------------------------------------------------------------------ Part B
    print("\n" + "=" * 88)
    print("Part B: kernel-only CUPTI us/call; reg NSTAGE sweep vs base and FLA dhu")
    print("=" * 88)
    shapes = [(1, 512, 4), (2, 1024, 8), (8, 2048, 8)]
    for (B, T, H) in shapes:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        F64 = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)  # operands for BT<=64
        for BT in (16, 32, 64):
            if BT == 64:
                fla_us = cuda_kernel_us(lambda: chunk_gated_delta_rule_bwd_dhu(
                    q=F64["fi"]["qg"], k=F64["fi"]["kg"], w=F64["fi"]["w"],
                    gk=F64["fi"]["g_cum"], h0=h0, dht=None, do=do, dv=F64["dv"],
                    scale=scale, chunk_size=64), "bwd_kernel_dhu")
                F = F64
            elif BT == 32:
                F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
                fla_us = float("nan")
            else:  # FLA intro kernel needs chunk_size >= 32; reuse BT=64 operands as labels
                F = F64
                fla_us = float("nan")
            ftag = f"{fla_us:7.1f}us" if fla_us == fla_us else "   n/a  "
            base_us = cuda_kernel_us(lambda F=F, BT=BT: run(
                lib_base, F["fi"], do, F["dv"], scale, BT, 1, "b1_launch"), "b1_dhu_cuda")
            print(f"  B{B} T{T} H{H}  BT={BT}   FLA dhu = {ftag}   base(wmma,NS=1) = {base_us:7.1f}us")
            for ns in (1, 2, 3):
                try:
                    us = cuda_kernel_us(lambda F=F, BT=BT, ns=ns: run(
                        lib_reg, F["fi"], do, F["dv"], scale, BT, ns), "b1_reg_cuda")
                except RuntimeError as e:
                    print(f"      NSTAGE={ns}   smem-exceeded ({e})")
                    continue
                xtag = f"xFLA={us/fla_us:5.2f}" if fla_us == fla_us else "xFLA=  n/a"
                print(f"      reg NSTAGE={ns}   {us:8.1f}us   {xtag}  xbase={us/base_us:5.2f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
