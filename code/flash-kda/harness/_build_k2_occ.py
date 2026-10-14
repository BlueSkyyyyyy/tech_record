"""Build FlashKDA K2 variants for the occupancy experiment.

Upstream K2: InputStages=3, OutputStages=2 -> 98,432 B smem/block -> 2 CTA/SM
on H100 (228 KB smem/SM).  The smem budget is

    sizeof(SharedStorageK2) = 32768 (bf16 state_acc, D*D*2)
                            + max(IN*17984 + OUT*4096, 65536)   <- union
                            + ~128 (pipelines + barrier)

The union's second arm is an *unconditional* fp32 state-conversion buffer of
65536 B (D*D*4), used only when StateFP32=true.  For the common bf16 path it is
dead weight and pins the union at 65536 -> total 98,432 regardless of stage
counts (as long as IN*17984 + OUT*4096 <= 65536).

Two knobs are exposed:
  * KDA_IN_STAGES / KDA_OUT_STAGES  -- shrink the TMA pipelines.
  * KDA_COND_FP32 -- make the fp32 state buffer size 1 when StateFP32=false
    (template `ReserveFP32 = StateFP32`), so the bf16 path's union is only the
    pipeline arrays.

Usage:
  python _build_k2_occ.py --in 2 --out 2 --cond 1 --name kda_occ_c_in2_out2
"""
import argparse
import os
import re
import shutil

UP = os.path.expanduser("~/github/FlashKDA")
BASE = "/tmp/kda_occ"


def prepare_src(cond):
    work = os.path.join(BASE, "src_cond" if cond else "src_plain")
    src = os.path.join(work, "csrc")
    if not os.path.isdir(src):
        os.makedirs(work, exist_ok=True)
        shutil.copytree(os.path.join(UP, "csrc"), src)
    lc = os.path.join(src, "smxx", "fwd_launch.cu")
    txt = open(lc).read()
    txt = re.sub(r"constexpr int kInputStages = \d+;",
                 "constexpr int kInputStages = KDA_IN_STAGES;", txt)
    txt = re.sub(r"constexpr int kOutputStages = \d+;",
                 "constexpr int kOutputStages = KDA_OUT_STAGES;", txt)
    if "#ifndef KDA_IN_STAGES" not in txt:
        txt = txt.replace(
            '#include "fwd_kernel2.cuh"',
            '#include "fwd_kernel2.cuh"\n\n'
            '#ifndef KDA_IN_STAGES\n#define KDA_IN_STAGES 3\n#endif\n'
            '#ifndef KDA_OUT_STAGES\n#define KDA_OUT_STAGES 2\n#endif')
    if cond:
        txt = txt.replace(
            "SharedStorageK2<K2L, kInputStages, kOutputStages>",
            "SharedStorageK2<K2L, kInputStages, kOutputStages, StateFP32>")
    open(lc, "w").write(txt)

    if cond:
        k2 = os.path.join(src, "smxx", "fwd_kernel2.cuh")
        t = open(k2).read()
        t = t.replace(
            "template <class Layouts, int InputStages, int OutputStages>\nstruct SharedStorageK2 {",
            "template <class Layouts, int InputStages, int OutputStages, bool ReserveFP32 = true>\nstruct SharedStorageK2 {")
        t = t.replace(
            "        alignas(128) char state_fp32_buf[cute::cosize_v<StateSmemLayout> * sizeof(float)];",
            "        alignas(128) char state_fp32_buf[ReserveFP32 ? cute::cosize_v<StateSmemLayout> * sizeof(float) : 1];")
        t = t.replace(
            "using SharedStorageT = SharedStorageK2<Layouts, InputStages, OutputStages>;",
            "using SharedStorageT = SharedStorageK2<Layouts, InputStages, OutputStages, StateFP32>;")
        assert "ReserveFP32 = true" in t and "ReserveFP32 ? cute" in t, "kernel patch failed"
        assert "OutputStages, StateFP32>;" in t, "kernel SharedStorageT patch failed"
        open(k2, "w").write(t)
    return src


def build(in_stages, out_stages, cond, name):
    from torch.utils.cpp_extension import load
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    src = prepare_src(cond)
    cutlass = os.path.join(UP, "cutlass")
    flags = [
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
        "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills",
        "-lineinfo", "--threads", "32",
        f"-DKDA_IN_STAGES={in_stages}", f"-DKDA_OUT_STAGES={out_stages}",
    ]
    mod = load(
        name=name,
        sources=[os.path.join(src, "flash_kda.cpp"),
                 os.path.join(src, "smxx", "fwd_launch.cu")],
        extra_include_paths=[
            os.path.join(cutlass, "include"),
            os.path.join(cutlass, "examples", "common"),
            os.path.join(cutlass, "tools", "util", "include"),
            src,
        ],
        extra_cflags=["-O3", "-Wno-psabi"],
        extra_cuda_cflags=flags,
        verbose=False,
    )
    print(f"BUILD_OK name={name} in={in_stages} out={out_stages} cond={cond}", flush=True)
    return mod


VARIANTS = {  # name -> (in, out, cond)
    "kda_occ_plain_in3_out2": (3, 2, 0),
    "kda_occ_plain_in2_out1": (2, 1, 0),
    "kda_occ_cond_in3_out2":  (3, 2, 1),
    "kda_occ_cond_in2_out2":  (2, 2, 1),
    "kda_occ_cond_in2_out1":  (2, 1, 1),
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_stages", type=int)
    ap.add_argument("--out", dest="out_stages", type=int)
    ap.add_argument("--cond", type=int, default=0)
    ap.add_argument("--name")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        for name, (i, o, c) in VARIANTS.items():
            build(i, o, c, name)
    else:
        build(a.in_stages, a.out_stages, a.cond, a.name)
