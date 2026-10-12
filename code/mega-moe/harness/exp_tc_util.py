#!/usr/bin/env python3
"""P26-1: DeepGEMM 的 tensor-core 利用率控制 `set_tc_util` —— H100 实测。

背景（参考源码 ~/github/DeepGEMM，commit 057ca59）：
  csrc/apis/config.hpp:21-25          —— Python API `set_tc_util` / `get_tc_util`
  csrc/runtime/runtime.hpp:22         —— Runtime.tc_util（默认 0）
  csrc/runtime/runtime.hpp:93-100     —— set/get：断言 0<=util<=100；get 把 0 翻成 100
  csrc/jit_kernels/heuristics/config.hpp:22 —— GemmDesc.tc_util（所有 kernel 的 desc 里都有）
  csrc/jit_kernels/impls/sm100_bf16_gemm.hpp:70 —— 唯一把它当**模板实参**传进 kernel 的 host
  deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh:35,141-142,326-343
                                      —— 唯一真正**实现节流**的 kernel：clock64 空转

结论（本脚本要坐实的）：
  `set_tc_util` 是一个**全局 runtime 开关**，但只有 **SM100 的 BF16 主 kernel** 把它的值
  编进 kernel（`kTensorCoreUtilControl` 模板参数）并插入 `clock64` 空转节流；
  其它 kernel（SM90 bf16/fp8、SM100 fp8/fp4）虽然 desc 里带 `tc_util`，kernel 模板里**没有**它，
  所以 **H100 上改 `set_tc_util` 对性能/正确性/编译缓存都无影响**。下面是真机证据。

运行：bash harness/exp_tc_util.sh > harness/exp_tc_util.out.txt
"""
import sys
import time

import torch

try:
    import deep_gemm
except Exception as e:  # pragma: no cover
    print("FATAL: cannot import deep_gemm:", e)
    sys.exit(1)

DEV = "cuda"
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
deep_gemm.use_deterministic_algorithms(True)  # 关掉 cuBLASLt 回退，强制走自家 kernel

print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
      f"cc={torch.cuda.get_device_capability(0)} | device_num_sms={deep_gemm.get_num_sms()}")
print("=" * 100)

# ---------------------------------------------------------------- 1. API 语义
print("\n[1] API 语义：set_tc_util / get_tc_util")
print(f"    默认 get_tc_util() = {deep_gemm.get_tc_util()}  (runtime.hpp:22 默认 0 → get 翻成 100)")
for v in (100, 75, 50, 25, 1):
    deep_gemm.set_tc_util(v)
    ok = deep_gemm.get_tc_util() == v
    print(f"    set_tc_util({v:3d}) -> get_tc_util() = {deep_gemm.get_tc_util():3d}  {'OK' if ok else 'MISMATCH'}")
deep_gemm.set_tc_util(0)
print(f"    set_tc_util(  0) -> get_tc_util() = {deep_gemm.get_tc_util():3d}  (0 = 关闭控制 = 100% 利用)")
for bad in (-1, 101):
    try:
        deep_gemm.set_tc_util(bad)
        print(f"    set_tc_util({bad}) -> 未报错  (BAD，期望断言)")
    except Exception as e:
        print(f"    set_tc_util({bad:3d}) -> 正确被拒: {type(e).__name__}: {str(e)[:70]}")


# ---------------------------------------------------------------- 计时工具
def bench_us(fn, warmup=30, iters=200):
    """每次调用前后 sync，取中位数（含 launch）；避免 PDL 相邻重叠。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def relerr(x, ref):
    return ((x.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)).item()


# ---------------------------------------------------------------- 2. SM90 BF16 vs tc_util
print("\n[2] SM90 BF16 主 kernel（H100 真实设备代码）在不同 tc_util 下的延迟与正确性")
print("    形状 m=n=k=4096；每档先 set_tc_util 再连续测两轮（第二轮看编译缓存）")
m = n = k = 4096
a = torch.randn((m, k), device=DEV, dtype=torch.bfloat16)
b = torch.randn((n, k), device=DEV, dtype=torch.bfloat16)
d = torch.empty((m, n), device=DEV, dtype=torch.bfloat16)
ref = (a.float() @ b.float().t()).to(torch.bfloat16)

print(f"    {'tc_util':>8} {'t_round1(us)':>13} {'t_round2(us)':>13} {'TFLOPS':>8} {'relerr':>10}")
results = {}
for util in (100, 50, 10, 100):
    deep_gemm.set_tc_util(util)
    deep_gemm.bf16_gemm_nt(a, b, d)  # 触发/命中 JIT
    torch.cuda.synchronize()
    t1 = bench_us(lambda: deep_gemm.bf16_gemm_nt(a, b, d), warmup=5, iters=80)
    t2 = bench_us(lambda: deep_gemm.bf16_gemm_nt(a, b, d))
    err = relerr(d, ref)
    tf = 2 * m * n * k / (t2 / 1e6) / 1e12
    results[util] = (t1, t2)
    print(f"    {util:8d} {t1:13.1f} {t2:13.1f} {tf:8.0f} {err:10.2e}")
deep_gemm.set_tc_util(0)

# ---------------------------------------------------------------- 3. SM90 FP8 vs tc_util
print("\n[3] SM90 FP8（默认 1D2D）在不同 tc_util 下的延迟（同一数学，换后端证明与 kernel 无关）")
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8  # noqa: E402
m = n = 4096
k = 4096
x = torch.randn((m, k), device=DEV, dtype=torch.bfloat16)
w = torch.randn((n, k), device=DEV, dtype=torch.bfloat16)
x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=False)
w_fp8, w_sf = per_block_cast_to_fp8(w, use_ue8m0=False)
out = torch.empty((m, n), device=DEV, dtype=torch.bfloat16)
print(f"    {'tc_util':>8} {'t(us)':>10} {'TFLOPS':>8}")
for util in (100, 50, 10):
    deep_gemm.set_tc_util(util)
    deep_gemm.fp8_gemm_nt((x_fp8, x_sf), (w_fp8, w_sf), out)
    t = bench_us(lambda: deep_gemm.fp8_gemm_nt((x_fp8, x_sf), (w_fp8, w_sf), out))
    tf = 2 * m * n * k / (t / 1e6) / 1e12
    print(f"    {util:8d} {t:10.1f} {tf:8.0f}")
deep_gemm.set_tc_util(0)

# ---------------------------------------------------------------- 4. 编译缓存
print("\n[4] 编译缓存：切换 tc_util 是否触发重编译？（微源码不含 tc_util → 应命中）")
m = n = k = 1024
a4 = torch.randn((m, k), device=DEV, dtype=torch.bfloat16)
b4 = torch.randn((n, k), device=DEV, dtype=torch.bfloat16)
d4 = torch.empty((m, n), device=DEV, dtype=torch.bfloat16)


def timed_call():
    t0 = time.time()
    deep_gemm.bf16_gemm_nt(a4, b4, d4)
    torch.cuda.synchronize()
    return time.time() - t0


deep_gemm.set_tc_util(100)
t_first = timed_call()
deep_gemm.set_tc_util(33)
t_second = timed_call()
deep_gemm.set_tc_util(0)
print(f"    首次调用（JIT 编译）      : {t_first*1e3:8.2f} ms")
print(f"    切换 tc_util 后再调用     : {t_second*1e3:8.2f} ms  "
      f"({'命中缓存' if t_second < t_first/3 else '疑似重编译'})")

# ---------------------------------------------------------------- 5. 公式复刻
print("\n[5] 节流公式复刻（kernel :337-338），确认常量与「降频意图」")
print("    kNumUMMACycles  = 2 * UMMA_M * UMMA_N * BLOCK_K / 8192   (推导的 UMMA 周期数)")
print("    kNumDummyCycles = (100 - util) * kNumUMMACycles / util   (要空转的周期数)")
print(f"    {'UMMA_M':>7} {'UMMA_N':>7} {'BLOCK_K':>8} {'util':>5} {'UMMAcyc':>10} {'dummycyc':>10}")
for (umn, unn, bk) in ((256, 128, 256), (128, 256, 256), (256, 64, 256)):
    cyc = 2 * umn * unn * bk // 8192
    for util in (100, 50, 25):
        dummy = (100 - util) * cyc // util
        print(f"    {umn:7d} {unn:7d} {bk:8d} {util:5d} {cyc:10d} {dummy:10d}")

print("\n" + "=" * 100)
print("小结：set_tc_util 在 H100（SM90 bf16/fp8）上只改 runtime 状态（与 0→100 语义），")
print("     不改 kernel、不影响性能/正确性/编译缓存——因为节流只编进 SM100 BF16 主 kernel。")
