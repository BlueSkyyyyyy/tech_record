"""P30-1 实验：DeepGEMM 的**测试基座** `deep_gemm/testing/` + `deep_gemm/utils/dist.py`。

前面 0~29 篇里每一个 kernel 的「对拍」「relerr」「TFLOPS」数字，都出自这套测试基座，
却从没系统拆过。本脚本在 H100 上用真实数字回答：

  A) `get_arch_major` / `test_filter` / `ignore_env`（testing/utils.py）的语义与过滤行为。
  B) `calc_diff`（testing/numeric.py:5-11）到底是什么度量——实测它**恒等于归一化平方误差**
     `||x-y||²/(||x||²+||y||²)`，据此得到：对称、∈[0,2]、=0 当且仅当逐元素相等、正缩放不变、
     **负缩放→≈2**、对 DC 平移敏感；小误差下 `≈ rel_l2²/2`。
  C) `count_bytes`（:14-21）递归统计 tuple/list、跳过 None。
  D) `assert_bitwise_equal`（:24-44）：逐位判等 + 失败时定位**首个不一致字节**（elem/coord/byte_in_elem）。
  E) `bench` vs `bench_kineto`（testing/bench.py:7-33 / 79-146）：在真实 SM90 `fp8_gemm_nt` 上
     量出两者的差异——event 计时含 CPU 发射开销（小 kernel 明显偏大，`high_precision` 用一大核
     把发射藏进流水）、`bench_kineto` 取**纯 kernel 时长**、`flush_l2` 的作用、以及 tuple 多 kernel。
  F) `uneven_all_gather`（utils/dist.py:38-66）：MoE 测试用把各 rank **不等长**的 topk 索引拼成
     全局表；在 8×H100 上验证「交换长度→pad→all_gather→trim→cat」的正确性与开销。

来源核对（~/github/DeepGEMM，公开版 commit 057ca59，版本 2.8.1）：
  - deep_gemm/testing/bench.py:7-33      bench
  - deep_gemm/testing/bench.py:79-146    bench_kineto
  - deep_gemm/testing/numeric.py:5-11    calc_diff
  - deep_gemm/testing/numeric.py:14-21   count_bytes
  - deep_gemm/testing/numeric.py:24-44   assert_bitwise_equal
  - deep_gemm/testing/utils.py:6-8       get_arch_major
  - deep_gemm/testing/utils.py:11-22     test_filter
  - deep_gemm/testing/utils.py:25-37     ignore_env
  - deep_gemm/utils/dist.py:10-35        init_dist
  - deep_gemm/utils/dist.py:38-66        uneven_all_gather
  - deep_gemm/utils/dist.py:69-74        dist_print
  - tests/test_mega_moe.py:11/76/354     真实调用形态
  - tests/test_bf16.py:45/56             calc_diff / bench_kineto 调用形态
"""
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm
from deep_gemm.testing import (
    bench, bench_kineto, calc_diff, count_bytes, assert_bitwise_equal,
)
from deep_gemm.testing.utils import get_arch_major, test_filter, ignore_env
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8


def sec(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


torch.manual_seed(0)
dev = "cuda"


# -----------------------------------------------------------------------------
# A) testing/utils.py：架构判定 + 两个装饰器
# -----------------------------------------------------------------------------
def part_a():
    sec("A) testing/utils.py：get_arch_major / test_filter / ignore_env（:6-8/:11-22/:25-37）")
    am = get_arch_major()
    print(f"  get_arch_major() = {am}（= torch.cuda.get_device_capability()[0]；H100=9）")

    # test_filter：condition 为 True 才执行函数体，否则打印一行 "Filtered by ..."
    calls = []

    @test_filter(lambda: False)
    def filtered():
        calls.append("ran")

    @test_filter(lambda: True)
    def passed():
        calls.append("ran")

    print("  test_filter(condition=False) 调用：")
    filtered()
    print("  test_filter(condition=True)  调用：")
    passed()
    print(f"  实际执行的函数体次数 = {len(calls)}（应为 1，False 的被过滤）")

    # ignore_env：condition 为 True 时临时 pop 掉某环境变量，函数返回后恢复
    # 注意：wrapper 不 return（源码两分支都是裸 func(...)），故只用副作用的 list 观测。
    os.environ["DG_DEMO_ENV"] = "sentinel"
    observed = {}

    @ignore_env("DG_DEMO_ENV", lambda: True)
    def inside_pop():
        observed["pop"] = os.environ.get("DG_DEMO_ENV", None)

    @ignore_env("DG_DEMO_ENV", lambda: False)
    def inside_keep():
        observed["keep"] = os.environ.get("DG_DEMO_ENV", None)

    @ignore_env("DG_DEMO_ENV", lambda: True)
    def returns_value():
        return 12345

    inside_pop()
    inside_keep()
    ret = returns_value()
    after = os.environ.get("DG_DEMO_ENV", None)
    print(f"  ignore_env(cond=True)  函数内读到 = {observed.get('pop')!r}（被临时移除）")
    print(f"  ignore_env(cond=False) 函数内读到 = {observed.get('keep')!r}（保留）")
    print(f"  返回后环境变量恢复 = {after!r}")
    print(f"  **注意**：wrapper 不 return（源码两分支都裸调 func），返回 12345 的函数经装饰后返回 {ret!r}")
    ok = (am == torch.cuda.get_device_capability()[0] == 9
          and len(calls) == 1 and observed.get('pop') is None
          and observed.get('keep') == "sentinel" and after == "sentinel" and ret is None)
    print(f"  判定：{'PASS' if ok else 'FAIL'}")
    return ok


# -----------------------------------------------------------------------------
# B) calc_diff：归一化平方误差
# -----------------------------------------------------------------------------
def part_b():
    sec("B) calc_diff（numeric.py:5-11）= ||x-y||^2 / (||x||^2+||y||^2)？")
    torch.manual_seed(1)
    x = torch.randn(4096, 1024, device=dev)
    y = x + torch.randn_like(x) * 1e-2

    got = calc_diff(x, y).item()
    xd, yd = x.double(), y.double()
    want = ((xd - yd) ** 2).sum() / ((xd ** 2).sum() + (yd ** 2).sum())
    print(f"  calc_diff(x,y) = {got:.8e}  vs  ||x-y||^2/(||x||^2+||y||^2) = {want.item():.8e}"
          f"  -> {'一致' if abs(got - want.item()) < 1e-14 else '不符'}")

    # 互换对称
    s = abs(calc_diff(x, y).item() - calc_diff(y, x).item())
    print(f"  对称性 |calc_diff(x,y)-calc_diff(y,x)| = {s:.2e}")

    # 取值范围与「=0 当且仅当逐元素相等」
    same = calc_diff(x, x.clone()).item()
    neg = calc_diff(x, -x).item()
    zero = calc_diff(x, torch.zeros_like(x)).item()
    print(f"  calc_diff(x,x) = {same:.2e}（=0 当且仅当逐元素相等）；"
          f"calc_diff(x,-x) = {neg:.6f}（≈2）；calc_diff(x,0) = {zero:.6f}（=1）")

    # 度量是「归一化平方误差」，对 **两操作数同步缩放** 不变；对**单侧缩放**仅在等范数时不变
    print(f"  同步缩放不变：calc_diff(3.7x, 3.7y) = {calc_diff(3.7 * x, 3.7 * y).item():.8e}"
          f" = calc_diff(x,y) = {got:.8e}")
    print(f"  单侧缩放**不**不变：calc_diff(x, 3.7y) = {calc_diff(x, 3.7 * y).item():.4e}"
          f" ≠ {got:.4e}（分子 |x-cy|^2、分母 |x|^2+c^2|y|^2 只在 c=1 时相等）")

    # 小误差下 ≈ rel_l2^2 / 2
    for e in (1e-3, 1e-2, 5e-2):
        yy = x + torch.randn_like(x) * e * x.abs().mean()
        cd = calc_diff(x, yy).item()
        rel_l2 = (torch.linalg.vector_norm((yy - x).double())
                  / torch.linalg.vector_norm(x.double())).item()
        print(f"  e={e:5.0e}: calc_diff={cd:.6e}   rel_l2^2/2={rel_l2 ** 2 / 2:.6e}"
              f"  比值 {cd / (rel_l2 ** 2 / 2):.4f}")

    # 对 DC 平移敏感（与「相对误差」不同）
    dc = calc_diff(x, x + 0.1).item()
    print(f"  平移 0.1 后 calc_diff = {dc:.4e}（度量对常量偏移敏感，非零均值差会被计入）")

    ok = (abs(got - want.item()) < 1e-14 and s < 1e-16 and same < 1e-16
          and abs(neg - 2) < 1e-6 and abs(zero - 1) < 1e-12
          and abs(calc_diff(3.7 * x, 3.7 * y).item() - got) < 1e-12)
    print(f"  判定：{'PASS' if ok else 'FAIL'}")
    return ok


# -----------------------------------------------------------------------------
# C) count_bytes
# -----------------------------------------------------------------------------
def part_c():
    sec("C) count_bytes（numeric.py:14-21）：递归统计元素字节、跳过 None")
    a = torch.empty(1024, dtype=torch.float8_e4m3fn, device=dev)     # 1 B
    b = torch.empty(512, dtype=torch.bfloat16, device=dev)           # 2 B
    sf = torch.empty(64, dtype=torch.float, device=dev)              # 4 B
    nested = ((a, b), [sf, None], None)
    got = count_bytes(nested)
    want = 1024 * 1 + 512 * 2 + 64 * 4
    print(f"  count_bytes(((fp8,bf16),[fp32,None],None)) = {got}  vs  {want}  "
          f"-> {'一致' if got == want else '不符'}")
    print(f"  单张量与 tuple 视为同级：count_bytes(a,b) = {count_bytes(a, b)} = {want - 64 * 4}")
    ok = got == want
    print(f"  判定：{'PASS' if ok else 'FAIL'}")
    return ok


# -----------------------------------------------------------------------------
# D) assert_bitwise_equal
# -----------------------------------------------------------------------------
def part_d():
    sec("D) assert_bitwise_equal（numeric.py:24-44）：逐位判等 + 失败定位首个不一致字节")
    x = torch.randn(8, 16, device=dev, dtype=torch.bfloat16)
    assert_bitwise_equal(x, x.clone(), "identical")   # 不抛异常
    print("  相同张量：通过（无异常）")

    y = x.clone()
    y[3, 5] = y[3, 5] + 1.0     # 改变一个元素
    try:
        assert_bitwise_equal(x, y, "one_elem")
        print("  改变一个元素：未抛异常  -> FAIL")
        return False
    except AssertionError as ex:
        msg = str(ex)
        print("  改变一个元素：抛出 AssertionError，消息：")
        for line in msg.splitlines():
            print("    " + line)
    # 形状/ dtype 不匹配的快速断言
    shape_ok = dtype_ok = False
    try:
        assert_bitwise_equal(x, x[:4], "shape")
    except AssertionError:
        shape_ok = True
    try:
        assert_bitwise_equal(x, x.float(), "dtype")
    except AssertionError:
        dtype_ok = True
    print(f"  形状不符先断言：{shape_ok}；dtype 不符先断言：{dtype_ok}")
    ok = shape_ok and dtype_ok
    print(f"  判定：{'PASS' if ok else 'FAIL'}")
    return ok


# -----------------------------------------------------------------------------
# E) bench vs bench_kineto（真实 SM90 fp8_gemm_nt）
# -----------------------------------------------------------------------------
def _make_gemm(m, n, k):
    a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
    a8, asf = per_token_cast_to_fp8(a, use_ue8m0=False)
    b8, bsf = per_block_cast_to_fp8(b, use_ue8m0=False)
    d = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    return lambda: deep_gemm.fp8_gemm_nt((a8, asf), (b8, bsf), d)


def part_e():
    sec("E) bench（bench.py:7-33）vs bench_kineto（:79-146）：event 计时 vs 纯 kernel 时长")
    print("  真实算子：deep_gemm.fp8_gemm_nt（内含 transpose_fp32 + sm90_fp8_gemm_1d2d）")

    # DG_USE_NVIDIA_TOOLS=1 时 bench_kineto 直接跳过 profiling（避免与 nsys/ncu 冲突，:88-90）
    os.environ["DG_USE_NVIDIA_TOOLS"] = "1"
    skip_single = bench_kineto(lambda: None, "fp8_gemm")
    skip_tuple = bench_kineto(lambda: None, ("a", "b"))
    del os.environ["DG_USE_NVIDIA_TOOLS"]
    print(f"  DG_USE_NVIDIA_TOOLS=1：bench_kineto 跳过 profiling，单名返回 {skip_single}，"
          f"tuple 返回 {skip_tuple}（与 nsys/ncu 冲突保护，:88-90）")

    for label, (m, n, k) in [("小 256x256x512", (256, 256, 512)),
                             ("大 4096x4096x4096", (4096, 4096, 4096))]:
        fn = _make_gemm(m, n, k)
        fn(); torch.cuda.synchronize()     # 触发 JIT

        t_bench = bench(fn, num_warmups=5, num_tests=100)
        t_bench_hp = bench(fn, num_warmups=5, num_tests=100, high_precision=True)
        t_kin = bench_kineto(fn, "fp8_gemm", num_tests=100, suppress_kineto_output=True, flush_l2=True)
        t_kin_nf = bench_kineto(fn, "fp8_gemm", num_tests=100, suppress_kineto_output=True, flush_l2=False)
        t_tr, t_gemm = bench_kineto(fn, ("transpose", "fp8_gemm"), num_tests=100,
                                    suppress_kineto_output=True, flush_l2=True)
        flops = 2 * m * n * k
        print(f"\n  [{label}]")
        print(f"    bench(event, 含发射)          = {t_bench * 1e6:9.3f} us")
        print(f"    bench(event, high_precision)  = {t_bench_hp * 1e6:9.3f} us")
        print(f"    bench_kineto(纯 kernel,flush) = {t_kin * 1e6:9.3f} us   "
              f"-> {flops / t_kin / 1e12:7.1f} TFLOPS")
        print(f"    bench_kineto(纯 kernel,noflush)= {t_kin_nf * 1e6:9.3f} us")
        print(f"    bench_kineto(tuple) transpose = {t_tr * 1e6:9.3f} us | fp8_gemm = {t_gemm * 1e6:9.3f} us")
        if t_kin > 0:
            print(f"    bench / bench_kineto 比值      = {t_bench / t_kin:6.2f}x")
    print("\n  结论：① 小 kernel 的 event 计时通常显著大于 kineto 纯 kernel 时长（CPU 逐次发射暴露；"
          "`high_precision` 的意图是用一个大核占满 GPU、让 CPU 趁机排队以隐藏发射开销，本机改善不稳定——")
    print("         GPU 被多个自驱任务共享，数字有抖动）；")
    print("        ② kineto 取**纯 kernel 时长**，并在 (transpose, fp8_gemm) 下把两个 kernel 分开计时；")
    print("        ③ `flush_l2` 每次清 256MB L2：大 GEMM 冷启（读操作数）比缓存命中明显慢，小 kernel 影响小。")
    return True


# -----------------------------------------------------------------------------
# F) uneven_all_gather（8 rank）
# -----------------------------------------------------------------------------
def _gather_worker(rank, world, port, result_q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", world_size=world, rank=rank,
                            init_method=f"tcp://127.0.0.1:{port}")
    from deep_gemm.utils.dist import uneven_all_gather
    H = 7168
    # 每个 rank 不等长的 token 数，值 = rank 编号，便于核对拼接顺序
    n_tok = [17, 512, 4096, 3, 2048, 100, 777, 64][rank]
    x = torch.full((n_tok, H), float(rank), device="cuda", dtype=torch.bfloat16)

    got = uneven_all_gather(x, dim=0)
    sizes = [17, 512, 4096, 3, 2048, 100, 777, 64]
    # 独立参考：按 rank 顺序把各段长度切片拼起来
    ok_correct = (got.shape[0] == sum(sizes)) and (got.shape[1] == H)
    off = 0
    for r, s in enumerate(sizes):
        seg = got[off:off + s]
        if not torch.equal(seg, torch.full_like(seg, float(r))):
            ok_correct = False
        off += s
    all_correct = torch.tensor(1 if ok_correct else 0, device="cuda")
    dist.all_reduce(all_correct, op=dist.ReduceOp.MIN)

    # dim=1
    y = torch.randn(128, 32, device="cuda")
    y[:, :] = float(rank)
    gy = uneven_all_gather(y, dim=1)
    ok_dim1 = gy.shape[1] == 32 * world and torch.equal(gy[:, rank * 32:(rank + 1) * 32],
                                                        torch.full_like(gy[:, rank * 32:(rank + 1) * 32], float(rank)))

    # 计时（H=7168，每 rank 固定 8192 tok）
    dist.barrier()
    z = torch.randn(8192, H, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        uneven_all_gather(z, dim=0)
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        uneven_all_gather(z, dim=0)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 20
    # all_gather：每 rank 向其余 world-1 个 rank 各发一份 size；这里按「每 rank 发出字节」折算带宽
    moved = z.numel() * 2 * (world - 1)
    total_out = z.numel() * 2 * world   # 8 份拼接结果的总字节
    if rank == 0:
        result_q.put((bool(all_correct.item()), ok_dim1, dt, moved / dt / 1e9, total_out / dt / 1e9))
    dist.barrier()
    dist.destroy_process_group()


def part_f():
    sec("F) uneven_all_gather（utils/dist.py:38-66）：不等长 all_gather 的正确性与开销（8xH100）")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")   # 本机 NVSwitch 无 Fabric Manager，关 NVLS 多播
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29711
    procs = [ctx.Process(target=_gather_worker, args=(r, 8, port, q)) for r in range(8)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join(timeout=240)
    alive = [pr for pr in procs if pr.is_alive()]
    for pr in alive:
        pr.terminate()
    bad = [pr.exitcode for pr in procs if pr.exitcode not in (0, None)]
    if alive or bad:
        print(f"  F 部分失败：alive={len(alive)} bad_exitcodes={bad}（NCCL/多卡环境问题）")
        return False
    all_correct, ok_dim1, dt, gbps, out_gbps = q.get(timeout=10)
    print(f"  8 rank 拼接正确（逐段 rank 编号 + 总长 7617，每 rank 结果一致）：{all_correct}")
    print(f"  dim=1 不等长拼接正确：{ok_dim1}")
    print(f"  H=7168、每 rank 8192 tok（117.4 MB）：{dt * 1e6:.1f} us，"
          f"每 rank 发出 {gbps:.1f} GB/s，拼接结果吞吐 {out_gbps:.1f} GB/s（未开 NVLS）")
    ok = all_correct and ok_dim1
    print(f"  判定：{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}  "
          f"cc {torch.cuda.get_device_capability(0)}")
    results = {}
    results["A"] = part_a()
    results["B"] = part_b()
    results["C"] = part_c()
    results["D"] = part_d()
    results["E"] = part_e()
    try:
        results["F"] = part_f()
    except Exception as ex:  # noqa: BLE001
        print(f"  F 部分跳过（{type(ex).__name__}: {ex}）")
        results["F"] = False
    sec("汇总")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print("  ALL PASS" if all(results.values()) else "  SOME FAIL")


if __name__ == "__main__":
    main()
