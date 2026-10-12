#!/usr/bin/env python3
"""P23-1: DeepGEMM 的 DeepJIT 运行时 —— 编译 / 缓存 / 加载 / 发射管线精读 + H100 实测。

参考源码（~/github/DeepGEMM commit 057ca59 + third-party/deep_jit commit 2efdab4）：
  csrc/runtime/jit.hpp:13-33              —— DeepGEMM 的 init_jit：Config(prefix="DG",
                                             tag="cutlass-<ver>", include=<pkg>/include,
                                             prefixes={"deep_gemm/"})，追加 nvcc flags
  deep_gemm/__init__.py:122               —— import 时 _C.init(包目录)
  csrc/python_api.cpp:31                  —— deep_jit::register_python_api(m, jit)
  third-party/deep_jit/include/deep_jit/runtime/runtime.hpp:19-98
                                          —— Runtime：mem/disk cache、hash_base 五段、compile/cache_key
  .../runtime/config.hpp:12-45            —— Config 校验
  .../cache/disk.hpp:79-134               —— DiskCache：<root>/cache/<tag>.<digest>、多根、".committed"
  .../cache/disk.hpp:26-74                —— DiskCacheEntry：原子 rename 发布 + 失败清理
  .../cache/memory.hpp:9-22               —— MemCache（进程内，key=cache_key）
  .../utils/hash.hpp:10-42                —— FNV1a：两 64bit state、长度前缀、splitmix 输出 hex
  .../utils/parser.hpp:37-120             —— 递归 include 哈希（只跟 include_prefixes）
  .../backend/cuda/backend.hpp:77-168     —— nvcc --cubin 编译 + 溢出/本地内存检查 + PTX/SASS dump
  .../backend/cuda/backend.hpp:174-217    —— nvcc/cuobjdump 定位（CUDA_HOME→which→/usr/local/cuda）
  .../backend/cuda/kernel.hpp:48-171      —— cuLibraryLoadFromFile + cuLaunchKernelEx
  .../backend/cuda/options.hpp:39-156     —— CompilerOptions 默认/合并/get_flags/update_hash
  .../backend/cuda/device.hpp:68-80       —— arch 字符串 sm_90a / sm_100f
  .../utils/env.hpp:56-84                 —— 环境变量前缀回退到全局 "DJ"
  .../python_api.hpp:12-20                —— get_jit() 暴露 Runtime

本脚本（全部 H100 可跑，主机侧真实执行）：
  1. 编译真机 kernel，解剖磁盘缓存条目（kernel.cu / kernel.cubin / meta.json / .committed）。
  2. **纯 Python 逐位复刻 DeepJIT 的 cache key**，与磁盘目录名比对（五段 FNV1a 哈希）。
  3. 首次编译 vs 缓存命中的耗时对比（跨进程）。
  4. 抓真实 nvcc 命令行与 flags（DG_JIT_PRINT_COMPILER_COMMAND / DG_JIT_DEBUG）。
  5. digest 敏感性：m 不进 key（compiled_dims="nk"），n/k、major、block config 进 key。
  6. DG_JIT_CHECK_NO_SPILLS 在真 kernel 上生效；ptxas 寄存器/溢出报告。
  7. FNV1a 算法性质（长度前缀消歧、双 state）。

运行：bash harness/exp_deepjit.sh > harness/exp_deepjit.out.txt
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

WORK = "/tmp/dg_jit_exp"
# DeepJIT 的 DiskCache 在读环境变量时是「惰性」的（Runtime 首次使用时才构造），
# 所以必须在任何 deep_gemm kernel 调用之前设置 DG_JIT_CACHE_DIR。
os.environ["DG_JIT_CACHE_DIR"] = os.path.join(WORK, "main")

import torch

try:
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8
except Exception as e:  # pragma: no cover
    print("FATAL: cannot import deep_gemm:", e)
    sys.exit(1)

MASK = (1 << 64) - 1


# ---------------------------------------------------------------------------
# 纯 Python 复刻 deep_jit::hash::FNV1a（utils/hash.hpp）
# ---------------------------------------------------------------------------
class FNV1a:
    def __init__(self):
        self.a = 0xc6a4a7935bd1e995
        self.b = 0xff51afd7ed558ccd

    def update(self, data):
        if isinstance(data, str):
            data = data.encode()
        n = len(data)
        prefix = bytes((n >> (8 * i)) & 0xFF for i in range(8))  # 长度前缀（小端 8B）
        for byte in prefix + data:
            self.a = ((self.a ^ byte) * 0x100000001B3) & MASK
            self.b = ((self.b ^ byte) * 0x9E3779B97F4A7C15) & MASK
        return self

    def digest(self):
        def sm(v):
            v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & MASK
            v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & MASK
            return v ^ (v >> 31)

        return "%016x%016x" % (sm(self.a), sm(self.b))


# 复刻 deep_jit::Parser（utils/parser.hpp）
def _trim_left(line):
    i = 0
    while i < len(line) and line[i] in " \t\f\v":
        i += 1
    return line[i:]


def _parse_include(line, prefixes):
    line = _trim_left(line)
    if not line.startswith("#"):
        return ""
    line = _trim_left(line[1:])
    if not line.startswith("include"):
        return ""
    line = _trim_left(line[len("include"):])
    if not line or line[0] not in "<\"":
        return ""
    closing = -1
    for i in range(1, len(line)):
        if line[i] in ">\"":
            closing = i
            break
    if closing < 0:
        return ""
    filename = line[1:closing]
    if line[0] != "<" or line[closing] != ">" or not filename:
        raise RuntimeError("non-standard include: %r" % line)
    for p in prefixes:
        if filename.startswith(p):
            return filename
    return ""


class Parser:
    def __init__(self, include_dirs, prefixes):
        self.include_dirs = include_dirs
        self.prefixes = prefixes
        self.cache = {}

    def parse_into_hash(self, code):
        h = FNV1a().update(code)
        for line in code.split(b"\n"):
            fn = _parse_include(line.decode("latin-1"), self.prefixes)
            if fn:
                h.update(self._includes_into_hash(fn))
        return h.digest()

    def _includes_into_hash(self, filename):
        if filename in self.cache:
            return self.cache[filename]
        path = None
        for d in self.include_dirs:
            cand = os.path.join(d, filename)
            if os.path.exists(cand):
                path = cand
                break
        if path is None:
            raise RuntimeError("include not found: %s" % filename)
        with open(path, "rb") as f:
            code = f.read()
        self.cache[filename] = self.parse_into_hash(code)
        return self.cache[filename]


def reconstruct_flags(opts):
    """复刻 cuda::CompilerOptions::get_flags（options.hpp:95-134）。"""
    flags = ["--gpu-architecture=sm_" + opts["arch"]]
    ol = opts["optimize_level"]
    flags += ["-O" + ol, "--compiler-options=-O" + ol]
    if opts["use_fast_math"]:
        flags.append("--use_fast_math")
    if opts["ptxas_verbose"]:
        flags.append("--ptxas-options=--verbose")
    flags.append("--ptxas-options=--register-usage-level=%d" % opts["ptxas_register_usage_level"])
    if opts["check_no_spills"]:
        flags.append("--ptxas-options=--warn-on-spills")
    if opts["check_no_local_memory"]:
        flags.append("--ptxas-options=--warn-on-local-memory-usage")
    if opts["with_line_info"]:
        flags.append("--generate-line-info")
    flags += opts["nvcc_flags"]
    flags += opts["extra_nvcc_flags"]
    return flags


def cache_key_from_entry(entry_dir):
    """从 meta.json + kernel.cu 复刻 deep_jit::Runtime::cache_key（runtime.hpp:46-53,92-98）。"""
    meta = json.load(open(os.path.join(entry_dir, "meta.json")))
    with open(os.path.join(entry_dir, "kernel.cu"), "rb") as f:
        source = f.read()
    # hash_base = extra_signature + compiler_info hash（nvcc --version 原文，含行尾）
    compiler_hash = FNV1a().update(meta["compiler_info"]["version"]).digest()
    base = FNV1a().update(meta["config"]["extra_signature"]).update(compiler_hash)
    # + flags + post_hook("") + parser.parse_into_hash(source)
    base.update(" ".join(reconstruct_flags(meta["compiler_options"])))
    base.update("")
    parser = Parser(meta["config"]["include_dirs"], meta["config"]["include_prefixes"])
    base.update(parser.parse_into_hash(source))
    return base.digest(), meta, parser


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------
def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def make_inputs(m, n, k, major_k=True, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    if not major_k:  # NN: B 列主序
        b = b.T.contiguous().T
    a_fp8, a_sf = per_token_cast_to_fp8(a, use_ue8m0=False)
    b_fp8, b_sf = per_block_cast_to_fp8(b, use_ue8m0=False)
    d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    return (a_fp8, a_sf), (b_fp8, b_sf), d


def run_in_subprocess(code, cache_dir, extra_env=None):
    env = dict(os.environ)
    env["DG_JIT_CACHE_DIR"] = cache_dir
    if extra_env:
        env.update(extra_env)
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True)
    dt = time.perf_counter() - t0
    return proc, dt


# ---------------------------------------------------------------------------
def main():
    print("torch", torch.__version__, "| cuda", torch.version.cuda)
    print("deep_gemm", os.path.dirname(deep_gemm.__file__))
    print("gpu", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    print("nvcc", subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1])

    work = WORK
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)

    # -----------------------------------------------------------------
    hr("§1 磁盘缓存条目解剖（编译一个真 kernel：fp8_gemm_nt m=1024 n=512 k=1024）")
    cache1 = os.path.join(work, "main")
    (a, b, d) = make_inputs(1024, 512, 1024)
    deep_gemm.fp8_gemm_nt(a, b, d)
    torch.cuda.synchronize()
    entries = sorted(glob.glob(os.path.join(cache1, "cache", "*")))
    print("磁盘缓存目录:", os.path.join(cache1, "cache"))
    for e in entries:
        print("  条目:", os.path.basename(e))
        print("    文件:", sorted(os.listdir(e)))
    # 详细展示 gemm 条目
    gemm_entry = [e for e in entries if "sm90_fp8_gemm_1d2d" in e]
    if gemm_entry:
        e = gemm_entry[0]
        src = open(os.path.join(e, "kernel.cu")).read()
        print("\n[生成源码 kernel.cu]（%d 字节）" % len(src))
        print(src)
        key, meta, parser = cache_key_from_entry(e)
        print("[缓存 key 复刻]")
        print("  磁盘目录 digest :", os.path.basename(e).split(".")[-1])
        print("  Python 复刻 digest:", key)
        print("  MATCH :", key == os.path.basename(e).split(".")[-1])
        print("  extra_signature :", meta["config"]["extra_signature"])
        print("  递归哈希的 deep_gemm/ 头文件数 :", len(parser.cache))
        print("  cache tag/prefix:", meta["config"]["include_prefixes"])
        print("  nvcc 命令行:")
        print("   ", meta["command"])

    # -----------------------------------------------------------------
    hr("§2 五段式 cache key（runtime.hpp:46-53 / 92-98）逐段拆解")
    if gemm_entry:
        key, meta, parser = cache_key_from_entry(e)
        compiler_hash = FNV1a().update(meta["compiler_info"]["version"]).digest()
        flags = reconstruct_flags(meta["compiler_options"])
        print("① hash_base 起点: extra_signature =", repr(meta["config"]["extra_signature"]))
        print("② hash_base 续: compiler_info hash = FNV1a(nvcc --version 原文)")
        print("   nvcc --version 原文以换行结尾 ->", repr(meta["compiler_info"]["version"][-20:]))
        print("   compiler_info hash =", compiler_hash)
        print("③ options.get_flags() 拼接（%d 个 flag，空格连接）:" % len(flags))
        for f in flags:
            print("     ", f)
        print("④ post_hook_hash = '' （无 post hook）")
        print("⑤ parser.parse_into_hash(source)（先哈希源码，再逐行追 deep_gemm/ include 递归哈希）")
        print("=> final digest =", key, "(与磁盘目录名一致:", key == os.path.basename(e).split(".")[-1], ")")

    # -----------------------------------------------------------------
    hr("§3 首次编译（cache miss）vs 缓存命中（跨进程）耗时")
    cache3 = os.path.join(work, "c3")
    snippet = (
        "import time,torch,deep_gemm\n"
        "from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8 as pb\n"
        "torch.manual_seed(0); m,n,k=2048,1024,2048\n"
        "a=torch.randn(m,k,device='cuda',dtype=torch.bfloat16); b=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)\n"
        "af,asf=per_token_cast_to_fp8(a,use_ue8m0=False); bf,bsf=pb(b,use_ue8m0=False)\n"
        "d=torch.empty(m,n,device='cuda',dtype=torch.bfloat16)\n"
        "t=time.perf_counter()\n"
        "deep_gemm.fp8_gemm_nt((af,asf),(bf,bsf),d); torch.cuda.synchronize()\n"
        "print('JIT_SECS %.3f'%(time.perf_counter()-t))\n"
    )
    p1, t1 = run_in_subprocess(snippet, cache3)
    jit1 = next((l for l in p1.stdout.splitlines() if l.startswith("JIT_SECS")), "?")
    print("第一次进程（miss，JIT 编译+加载两个 kernel）: %.2f s  rc=%d  %s" % (t1, p1.returncode, jit1))
    p2, t2 = run_in_subprocess(snippet, cache3)
    jit2 = next((l for l in p2.stdout.splitlines() if l.startswith("JIT_SECS")), "?")
    print("第二次进程（hit，从磁盘加载 cubin）: %.2f s  rc=%d  %s" % (t2, p2.returncode, jit2))
    n_entries = len(glob.glob(os.path.join(cache3, "cache", "*")))
    print("缓存条目数（第二次不应新增）:", n_entries)
    if p1.returncode != 0:
        print(p1.stderr[-2000:])
    # 用 DG_JIT_PRINT_LOAD_TIME 单独量加载时间
    p3, t3 = run_in_subprocess(snippet, cache3, {"DG_JIT_PRINT_LOAD_TIME": "1"})
    for line in p3.stdout.splitlines():
        if "Load time" in line:
            print("  ", line)

    # -----------------------------------------------------------------
    hr("§4 真实 nvcc 命令行 / flags 抓取（DG_JIT_PRINT_COMPILER_COMMAND=1）")
    cache4 = os.path.join(work, "c4")
    p4, t4 = run_in_subprocess(snippet, cache4, {"DG_JIT_PRINT_COMPILER_COMMAND": "1"})
    for line in p4.stdout.splitlines():
        if "Running command" in line or "nvcc" in line:
            print(line)

    hr("§4b DG_JIT_DEBUG=1：dump PTX/SASS + 加载耗时 + ptxas 报告")
    cache4b = os.path.join(work, "c4b")
    p4b, t4b = run_in_subprocess(snippet, cache4b, {"DG_JIT_DEBUG": "1"})
    interesting = [l for l in p4b.stdout.splitlines()
                   if any(s in l for s in ("Loading CUBIN", "Load time", "Running command",
                                           "Used", "bytes", "registers", "spill"))]
    for line in interesting[:20]:
        print(line)
    dumped = []
    for e in glob.glob(os.path.join(cache4b, "cache", "sm90_fp8_gemm*")):
        dumped = sorted(os.listdir(e))
    print("debug 编译产物目录:", dumped)

    # -----------------------------------------------------------------
    hr("§5 digest 敏感性：cache key 覆盖「完全实例化后的模板」（编译维度 + 启发式 block config）")
    cache5 = os.path.join(work, "c5")
    shapes = [(1024, 512, 1024), (3072, 512, 1024), (1024, 1024, 1024), (1024, 512, 2048)]
    body = "\n".join(
        "af,asf=per_token_cast_to_fp8(torch.randn(%d,%d,device='cuda',dtype=torch.bfloat16),use_ue8m0=False)\n"
        "bf,bsf=pb(torch.randn(%d,%d,device='cuda',dtype=torch.bfloat16),use_ue8m0=False)\n"
        "deep_gemm.fp8_gemm_nt((af,asf),(bf,bsf),torch.empty(%d,%d,device='cuda',dtype=torch.bfloat16))\n"
        % (m, k, n, k, m, n)
        for (m, n, k) in shapes
    )
    code5 = (
        "import torch,deep_gemm\n"
        "from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8 as pb\n"
        + body + "torch.cuda.synchronize()\n"
    )
    p5, _ = run_in_subprocess(code5, cache5)
    entries5 = sorted(glob.glob(os.path.join(cache5, "cache", "sm90_fp8_gemm*")))
    print("形状列表 (m,n,k):")
    for s in shapes:
        print("   ", s)
    print("产生的 gemm 磁盘条目数（每个不同的实例化模板一个条目）:", len(entries5))
    def template_block(e):
        src = open(os.path.join(e, "kernel.cu")).read()
        lines = src.splitlines()
        i = next(i for i, l in enumerate(lines) if "sm90_fp8_gemm_1d2d_impl" in l)
        return " ".join(l.strip() for l in lines[i + 1:i + 12]).replace("  ", " ")
    for e in entries5:
        print("  ", os.path.basename(e)[:34], "...  ", template_block(e))
    if p5.returncode != 0:
        print(p5.stderr[-1500:])
    # 忽略编译维度：set_ignore_compile_dims(True) → get_compiled_dim 一律返回 0
    cache5b = os.path.join(work, "c5b")
    code_ignore = ("import torch,deep_gemm\n"
                   "deep_gemm.set_ignore_compile_dims(True)\n"
                   "from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8 as pb\n"
                   + body + "torch.cuda.synchronize()\n")
    p5b, _ = run_in_subprocess(code_ignore, cache5b)
    entries5b = sorted(glob.glob(os.path.join(cache5b, "cache", "sm90_fp8_gemm*")))
    print("ignore_compile_dims=True 下同样 4 个形状的条目数:", len(entries5b))
    for e in entries5b:
        src = open(os.path.join(e, "kernel.cu")).read()
        dims = [l.strip() for l in src.splitlines() if l.strip() and l.strip()[0].isdigit()]
        print("  ", os.path.basename(e)[:40], "...  实例化维度行:", dims[0] if dims else "?")

    # -----------------------------------------------------------------
    hr("§6 DG_JIT_CHECK_NO_SPILLS：真 kernel 上强制无溢出（regex 检查 ptxas 输出）")
    cache6 = os.path.join(work, "c6")
    p6, _ = run_in_subprocess(snippet, cache6, {"DG_JIT_CHECK_NO_SPILLS": "1", "DG_JIT_PTXAS_VERBOSE": "1"})
    print("CHECK_NO_SPILLS=1 运行 rc =", p6.returncode, "（0 = 未检测到溢出）")
    spill_lines = [l for l in p6.stdout.splitlines() if "spill" in l.lower()]
    for l in spill_lines[:8]:
        print(l)
    if p6.returncode != 0:
        print(p6.stderr[-1500:])

    # -----------------------------------------------------------------
    hr("§7 FNV1a 算法性质（utils/hash.hpp:15-37）")
    h_ab_c = FNV1a().update("ab").update("c").digest()
    h_a_bc = FNV1a().update("a").update("bc").digest()
    print('FNV1a("ab"+"c") =', h_ab_c)
    print('FNV1a("a"+"bc") =', h_a_bc)
    print("长度前缀消歧 -> 两串不同:", h_ab_c != h_a_bc)
    print('FNV1a("") digest    =', FNV1a().update("").digest(), "（空串也参与，长度=0）")
    print("digest 固定 32 hex（两个 64bit state 各 splitmix 后拼接）:",
          len(FNV1a().update("x").digest()) == 32)

    hr("完成")


if __name__ == "__main__":
    main()
