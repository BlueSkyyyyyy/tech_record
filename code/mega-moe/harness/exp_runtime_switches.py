#!/usr/bin/env python3
"""P38-1 实验：DeepGEMM 的全局运行时/编译开关面（`csrc/apis/config.hpp` + `HeuristicsRuntime` +
`Runtime`）。全部架构无关，H100(sm90) 真机实测。

参考 ~/github/DeepGEMM：
  - csrc/apis/config.hpp            （pybind 开关面）
  - csrc/jit_kernels/heuristics/runtime.hpp （HeuristicsRuntime）
  - csrc/runtime/runtime.hpp        （Runtime：num_sms / tc_util / cuBLASLt）
  - csrc/apis/gemm.hpp:519          （deterministic ∧ is_cublaslt_available → 回退）
  - csrc/jit_kernels/impls/runtime_utils.hpp:31-40 （get_compiled_dim）
  - csrc/jit_kernels/heuristics/{sm90,sm100}.hpp   （block multiple / mk alignment）
  - csrc/jit_kernels/heuristics/mega_gate.hpp:111 / csrc/apis/mega_mhc.hpp:15 （batch invariance）
  - third-party/deep_jit/include/deep_jit/backend/cuda/kernel.hpp:148-151 （PDL launch attr）

输出：逐项 PASS/FAIL，末尾总计。
"""
import os
import re
import subprocess
import sys

import torch
import deep_gemm

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    tag = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{tag}] {name}" + (f"  :: {detail}" if detail else ""))


def profiled_kernel(fn, needle):
    """跑一次 fn，返回 name 含 needle 的 CUDA kernel 全名（含模板实参）。"""
    names = profiled_kernels(fn)
    hit = [n for n in names if needle in n]
    return hit[0] if hit else ""


def profiled_kernels(fn):
    """跑一次 fn，返回全部 CUDA kernel 名。"""
    from torch.profiler import profile, ProfilerActivity
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    cuda = torch.autograd.DeviceType.CUDA
    return [e.name for e in p.events() if e.device_type == cuda]


DEV_SMS = torch.cuda.get_device_properties(0).multi_processor_count


def bf16_nt(m=4096, n=4096, k=4096):
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    d = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    deep_gemm.bf16_gemm_nt(a, b, d)
    return d


print("=" * 78)
print("DeepGEMM runtime/compile switchboard — H100 (sm90) 真机")
print("=" * 78)
print(f"torch={torch.__version__}  deep_gemm={deep_gemm.__version__}  device_sms={DEV_SMS}")
print()

# ---------------------------------------------------------------- §1 num_sms
print("§1  set_num_sms / get_num_sms（Runtime，runtime.hpp:78-91）")
check("默认 num_sms == 设备 SM 数", deep_gemm.get_num_sms() == DEV_SMS,
      f"get_num_sms()={deep_gemm.get_num_sms()} dev={DEV_SMS}")
rejected = []
for bad in (0, -1, DEV_SMS + 1):
    try:
        deep_gemm.set_num_sms(bad)
        rejected.append(False)
    except Exception:
        rejected.append(True)
check("0 / -1 / >dev 越界被拒（:79 DG_HOST_ASSERT）", all(rejected), f"{rejected}")

deep_gemm.set_num_sms(64)
check("set_num_sms(64) 生效", deep_gemm.get_num_sms() == 64, f"={deep_gemm.get_num_sms()}")
# 关掉 cuBLASLt 回退后再抓 kernel 名（否则默认走 nvjet，看不到 deepgemm kernel）
deep_gemm.use_deterministic_algorithms(True)
kn64 = profiled_kernel(bf16_nt, "sm90_bf16_gemm_impl")
m = re.search(r"sm90_bf16_gemm_impl<([^>]*)>", kn64)
nums = re.findall(r"(\d+)u", m.group(1)) if m else []
check("kernel 模板实参含 num_sms=64（+ 强制走自研 kernel）",
      "sm90_bf16_gemm_impl" in kn64 and "64u" in m.group(1),
      f"模板=<{m.group(1)[:60]}...>" if m else "no kernel")
deep_gemm.set_num_sms(DEV_SMS)
kn132 = profiled_kernel(bf16_nt, "sm90_bf16_gemm_impl")
check("恢复 num_sms=dev 后模板回到 132u", "132u" in kn132, "")
deep_gemm.use_deterministic_algorithms(False)
print()

# ---------------------------------------------------------------- §2 tc_util
print("§2  set_tc_util / get_tc_util（Runtime，runtime.hpp:93-100）")
check("默认 get_tc_util()==100", deep_gemm.get_tc_util() == 100)
deep_gemm.set_tc_util(50)
check("set_tc_util(50) 生效", deep_gemm.get_tc_util() == 50)
deep_gemm.set_tc_util(0)
check("set_tc_util(0)==> getter 返回默认 100（:99）", deep_gemm.get_tc_util() == 100)
ok = []
for bad in (101, -1):
    try:
        deep_gemm.set_tc_util(bad)
        ok.append(False)
    except Exception:
        ok.append(True)
check("101 / -1 越界被拒（:94）", all(ok), f"{ok}")
print()

# ---------------------------------------------------------------- §3 pdl
print("§3  set_pdl / get_pdl（config.hpp:27-32 → deep_jit LaunchOptions.enable_pdl）")
p0 = deep_gemm.get_pdl()
check("get_pdl() 为布尔", isinstance(p0, bool), f"默认={p0}")
deep_gemm.set_pdl(True)
check("set_pdl(True) 回读 True", deep_gemm.get_pdl() is True)
deep_gemm.set_pdl(False)
check("set_pdl(False) 回读 False", deep_gemm.get_pdl() is False)
deep_gemm.set_pdl(p0)
print("  (PDL 落到 CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION，kernel.hpp:148-151；"
      "kernel 内用 cudaGridDependencySynchronize，如 sm90_bf16_gemm.cuh:131)")
print()

# -------------------------------------------------------- §4 mk alignment
print("§4  set/get_mk_alignment_for_contiguous_layout + theoretical（layout.hpp:133-141, runtime.hpp:57-64）")
mk0 = deep_gemm.get_mk_alignment_for_contiguous_layout()
th = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
check("默认 alignment == 128（SM90 legacy）", mk0 == 128, f"={mk0}")
check("theoretical(expected_m=None) == 128（非 arch10 恒 128）", th == 128, f"={th}")
deep_gemm.set_mk_alignment_for_contiguous_layout(256)
check("set_mk_alignment(256) 回读 256", deep_gemm.get_mk_alignment_for_contiguous_layout() == 256)
deep_gemm.set_mk_alignment_for_contiguous_layout(mk0)
check("恢复后回读 128", deep_gemm.get_mk_alignment_for_contiguous_layout() == 128)
print()

# -------------------------------------------------- §5 block size multiple
print("§5  set_block_size_multiple_of（config.hpp:39-47, sm90.hpp:40 / sm100.hpp:55,72）")
deep_gemm.set_block_size_multiple_of(64)          # int 形式
deep_gemm.set_block_size_multiple_of((64, 128))   # tuple 形式
check("int 与 tuple 两种入参均被接受", True, "(64) / (64,128)")
deep_gemm.set_block_size_multiple_of((1, 1))
print()

# ------------------------------------------------- §6 deterministic algos
print("§6  use_deterministic_algorithms：关掉 cuBLASLt 回退（gemm.hpp:519 / einsum.hpp:81,115）")
deep_gemm.use_deterministic_algorithms(False)
kn_nd = profiled_kernel(bf16_nt, "nvjet")
kn_nd2 = profiled_kernel(bf16_nt, "sm90_bf16_gemm_impl")
check("非确定性：bf16_gemm_nt 默认回退 cuBLASLt（nvjet）",
      kn_nd != "" and kn_nd2 == "", f"kernel={kn_nd or kn_nd2}")
deep_gemm.use_deterministic_algorithms(True)
kn_d = profiled_kernel(bf16_nt, "sm90_bf16_gemm_impl")
check("确定性：改走自研 SM90 kernel", kn_d != "", f"kernel={kn_d.split('<')[0]}")


def einsum_bhr():
    b, h, r, d = 4, 8, 512, 256
    A = torch.randn(b, h, r, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(h, d, r, dtype=torch.bfloat16, device="cuda")
    D = torch.empty(b, h, d, dtype=torch.bfloat16, device="cuda")
    deep_gemm.einsum("bhr,hdr->bhd", A, B, D)
    return D


kn_e_det = profiled_kernels(einsum_bhr)
own = [n for n in kn_e_det if ("bhr_hdr_bhd" in n or "gemm" in n)]
has_own = len(own) > 0
has_nvjet = any("nvjet" in n for n in kn_e_det)
check("确定性：einsum('bhr,hdr->bhd') 也走自研（非 nvjet）",
      has_own and not has_nvjet,
      f"kernel={own[0].split('<')[0] if own else kn_e_det}")
deep_gemm.use_deterministic_algorithms(False)
print()

# ------------------------------------------------ §7 DG_PRINT_CONFIGS env
print("§7  DG_PRINT_CONFIGS 环境变量（heuristics/common.hpp:44）——子进程抓配置")


def run_print_configs(extra_setup=""):
    code = f"""
import torch, deep_gemm
{extra_setup}
a=torch.randn(4096,4096,dtype=torch.bfloat16,device='cuda')
b=torch.randn(4096,4096,dtype=torch.bfloat16,device='cuda')
d=torch.empty(4096,4096,dtype=torch.bfloat16,device='cuda')
deep_gemm.bf16_gemm_nt(a,b,d); torch.cuda.synchronize()
"""
    env = dict(os.environ, DG_PRINT_CONFIGS="1")
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True).stdout
    line = next((l for l in out.splitlines() if "GemmConfig(layout=" in l), "")
    return line


def parse_layout(line):
    def g(pat):
        m = re.search(pat, line)
        return int(m.group(1)) if m else None
    return dict(
        block_m=g(r"block_m=(\d+)"), block_n=g(r"block_n=(\d+)"),
        num_sms=g(r"LaunchConfig\(num_sms=(\d+)"), tc_util=g(r"tc_util=(\d+)"),
    )


line_def = run_print_configs("deep_gemm.use_deterministic_algorithms(True)")
lay_def = parse_layout(line_def)
check("默认 block_n=128", lay_def["block_n"] == 128, f"{lay_def}")
line_256 = run_print_configs(
    "deep_gemm.use_deterministic_algorithms(True)\n"
    "deep_gemm.set_block_size_multiple_of((256,256))")
lay_256 = parse_layout(line_256)
check("block_size_multiple_of=(256,256) → block_n 变为 256 的倍数",
      lay_256["block_n"] == 256 and lay_256["block_n"] % 256 == 0, f"{lay_256}")
line_sms = run_print_configs(
    "deep_gemm.use_deterministic_algorithms(True)\n"
    "deep_gemm.set_num_sms(32)")
lay_sms = parse_layout(line_sms)
check("set_num_sms(32) → 打印的 LaunchConfig.num_sms=32", lay_sms["num_sms"] == 32, f"{lay_sms}")
print()

# ------------------------------------------- §8 ignore_compile_dims
print("§8  set_ignore_compile_dims（config.hpp:36-38, runtime_utils.hpp:31-40）——抓模板实参")


def tmpl_args(kn):
    m = re.search(r"sm90_bf16_gemm_impl<([^>]*)>", kn)
    return m.group(1) if m else ""


deep_gemm.use_deterministic_algorithms(True)
deep_gemm.set_ignore_compile_dims(False)
t_def = tmpl_args(profiled_kernel(bf16_nt, "sm90_bf16_gemm_impl"))
deep_gemm.set_ignore_compile_dims(True)
t_ign = tmpl_args(profiled_kernel(bf16_nt, "sm90_bf16_gemm_impl"))
deep_gemm.set_ignore_compile_dims(False)
check("默认 compiled_dims=nk → 模板 <M=0,N=4096,K=4096>",
      t_def.split(",")[2:5] == [" 0u", " 4096u", " 4096u"], f"{t_def[:50]}")
check("ignore_compile_dims → 模板 <0,0,0>（N/K 也置 0，runtime_utils.hpp:32-33）",
      t_ign.split(",")[2:5] == [" 0u", " 0u", " 0u"], f"{t_ign[:50]}")
deep_gemm.use_deterministic_algorithms(False)
print()

print("=" * 78)
print(f"TOTAL: {PASS} PASS / {FAIL} FAIL")
print("=" * 78)
sys.exit(1 if FAIL else 0)
