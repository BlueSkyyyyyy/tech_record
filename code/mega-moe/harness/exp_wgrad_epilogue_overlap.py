"""P138-1 实验：MoE wgrad 的 S 步归约能否「藏进 GEMM 的 HBM 余量」。

为什么做这篇（补的缺口）：
  第 147 篇（P137-1）把「wgrad 换 bf16 输出」落到端到端 1.67×（ALT-OPT），并断言 2.27× 不可达，
  因为「每步 bf16 梯度至少要**串行**读一遍（~76us/步）」。它留的口子（可继续①）是：
  若把归约/优化器更新**融进 GEMM 的 epilogue、或与下一步 GEMM 重叠**，这 76us 能不能藏掉？
  本篇回答它。

三个架构（字节账，G=E*M*N=1.17e8）：
  SEQ   (第 147 篇 ALT-OPT) : S 步各写 bf16(fresh 235MB/步) + **尾收**批量归约(读 S*235 + RMW fp32 470)
   INC-F : 每步写 bf16 + **增量**更新 fp32 master(读 g 235 + RMW fp32 470 = 1175MB/步)，**可与下一步 GEMM 重叠**
   INC-B : 每步写 bf16 + **增量**更新 bf16 acc(读 g 235 + RMW bf16 235 = 705MB/步)，**可重叠**、省显存

关键问题：增量式字节更多，但**可重叠**；批量式字节最少，但只能串行尾收。重叠能不能补回字节差？

本实验四部分：
  Part A · 消费侧内核（`src/wgrad_epilogue_overlap.cu`，架构无关）的字节/带宽 + SM 预算扫描
  Part B · 真机重叠：真实 SM90 GEMM（1D2D bf16）主流水线 ∥ 增量消费者侧流，扫 SM 预算，比串行/重叠
  Part C · 隔离控制：计算受限「伪 GEMM epilogue」生产者 ∥ 访存受限消费者（不依赖 DeepGEMM）
  Part D · 增量协议的数值：bf16 累加器 S 步的累积舍入

可跑性：全部 SM90 FP8 kernel + 架构无关消费核；H100(sm90) 真机。SM100 wgrad `tn` 同族 fp32 累积、仅分析。
本机 8 卡共享，本文在**空闲 GPU** 单进程测（可用 CUDA_VISIBLE_DEVICES 选卡）。

输出重定向：harness/exp_wgrad_epilogue_overlap.out.txt
"""
import ctypes
import os
import sys
import traceback

import torch

import deep_gemm

DG_DIR = os.environ.get('DEEPGEMM_DIR', os.path.expanduser('~/github/DeepGEMM'))
sys.path.insert(0, os.path.join(DG_DIR, 'tests'))

from deep_gemm.utils import (  # noqa: E402
    align, per_token_cast_to_fp8, per_channel_cast_to_fp8, per_block_cast_to_fp8,
)

GRAN = 128
DEV = 'cuda'
HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, 'wgrad_epilogue_overlap.so')

E, M, N, KE = 8, 2048, 7168, 512
G = E * M * N


class Lib:
    def __init__(self, path):
        c = ctypes.CDLL(path)
        self.c = c
        VP = ctypes.c_void_p
        c.woc_malloc_bf16.argtypes = [ctypes.c_long]; c.woc_malloc_bf16.restype = VP
        c.woc_malloc_f32.argtypes = [ctypes.c_long]; c.woc_malloc_f32.restype = VP
        c.woc_free.argtypes = [VP]
        for nm in ('woc_read', 'woc_acc_step', 'woc_fused_step', 'woc_fused_all', 'woc_gen'):
            pass
        c.woc_read.argtypes = [VP, ctypes.c_long, ctypes.c_int, VP, ctypes.c_int]; c.woc_read.restype = ctypes.c_double
        c.woc_acc_step.argtypes = [VP, VP, ctypes.c_long, ctypes.c_int, ctypes.c_int, VP, ctypes.c_int]; c.woc_acc_step.restype = ctypes.c_double
        c.woc_fused_step.argtypes = [VP, VP, ctypes.c_long, ctypes.c_int, ctypes.c_double, ctypes.c_int, VP, ctypes.c_int]; c.woc_fused_step.restype = ctypes.c_double
        c.woc_fused_all.argtypes = [VP, VP, ctypes.c_long, ctypes.c_int, ctypes.c_double, ctypes.c_int, VP, ctypes.c_int]; c.woc_fused_all.restype = ctypes.c_double
        c.woc_gen.argtypes = [VP, ctypes.c_long, ctypes.c_int, ctypes.c_int, VP, ctypes.c_int]; c.woc_gen.restype = ctypes.c_double
        c.woc_launch_acc.argtypes = [VP, VP, ctypes.c_long, ctypes.c_int, ctypes.c_int, VP]
        c.woc_launch_step.argtypes = [VP, VP, ctypes.c_long, ctypes.c_int, ctypes.c_double, ctypes.c_int, VP]
        c.woc_launch_all.argtypes = [VP, VP, ctypes.c_long, ctypes.c_int, ctypes.c_double, ctypes.c_int, VP]
        c.woc_launch_gen.argtypes = [VP, ctypes.c_long, ctypes.c_int, ctypes.c_int, VP]
        c.woc_gen_smem.argtypes = [VP, ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_int, VP, ctypes.c_int]; c.woc_gen_smem.restype = ctypes.c_double
        c.woc_launch_gen_smem.argtypes = [VP, ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_int, VP]
        c.woc_selftest.argtypes = [ctypes.POINTER(ctypes.c_double)]


def _bench(fn, warm=6, iters=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / 1e3 / iters


# --------------------------------------------------------------- operands (同 P137)
def dense_operands(Ag, Bg, host_ks):
    ops = []
    off = 0
    for k in host_ks:
        A = Ag[off:off + k].T.contiguous()
        B = Bg[off:off + k].T.contiguous()
        ops.append((per_token_cast_to_fp8(A, use_ue8m0=False, gran_k=GRAN),
                    per_block_cast_to_fp8(B, use_ue8m0=False, gran_k=GRAN)))
        off += k
    return ops


def build_operands(Smax, host_ks):
    sum_k = sum(host_ks)
    dns = []
    for _ in range(Smax):
        Ag = torch.randn(sum_k, M, device=DEV)
        Bg = torch.randn(sum_k, N, device=DEV)
        dns.append(dense_operands(Ag, Bg, host_ks))
        del Ag, Bg
    torch.cuda.empty_cache()
    return dns


# --------------------------------------------------------------- Part A
def part_a(lib, budget_default, n_sms):
    print('=' * 100)
    print('Part A · 消费侧内核（架构无关）字节/带宽 + SM 预算扫描')
    print('-' * 100)
    err = (ctypes.c_double * 3)()
    lib.c.woc_selftest(err)
    print(f'  正确性（GPU vs CPU double）：fused_step max|err|={err[1]:.3e}  fused_all max|err|={err[2]:.3e}'
          f'  -> {"OK" if max(err[1], err[2]) < 1e-2 else "FAIL"}')

    stack = torch.randn(16, G, device=DEV, dtype=torch.bfloat16)
    acc = torch.zeros(G, device=DEV, dtype=torch.bfloat16)
    w = torch.full((G,), 0.001, device=DEV, dtype=torch.float32)
    gp, ap, wp = stack.data_ptr(), acc.data_ptr(), w.data_ptr()

    tr = lib.c.woc_read(gp, G, budget_default, None, 30)
    print(f'  纯读一份 bf16 ({G * 2 / 1e6:.0f} MB)：{tr * 1e6:.1f} us  ({G * 2 / tr / 1e9:.2f} GB/s)')
    print(f'  {"核":<26}{"时(us)":>10}{"字节(MB)":>11}{"GB/s":>10}')
    ta = lib.c.woc_acc_step(gp, ap, G, 0, budget_default, None, 30)
    print(f'  {"acc_bf16 一步 (705MB)":<26}{ta * 1e6:>10.1f}{705:>11.0f}{705e6 / ta / 1e9:>10.2f}')
    ts = lib.c.woc_fused_step(gp, wp, G, 0, 0.25, budget_default, None, 30)
    print(f'  {"fused_step 一步 (1175MB)":<26}{ts * 1e6:>10.1f}{1175:>11.0f}{1175e6 / ts / 1e9:>10.2f}')
    for S in (1, 4, 8, 16):
        tb = lib.c.woc_fused_all(gp, wp, G, S, 0.25, budget_default, None, 20)
        by = S * 235 + 940
        print(f'  {"fused_all S=%d (%dMB)" % (S, by):<26}{tb * 1e6:>10.1f}{by:>11.0f}{by * 1e6 / tb / 1e9:>10.2f}')

    print(f'  -- 块数预算扫描（consumer = fused_step 一步；每步 1175MB；全带宽需 ~32 block/SM）--')
    print(f'  {"blocks":>8}{"时(us)":>10}{"GB/s":>10}')
    for b in (132, 264, 528, 1056, 2112, 4224, 8448):
        t = lib.c.woc_fused_step(gp, wp, G, 0, 0.25, b, None, 15)
        print(f'  {b:>8}{t * 1e6:>10.1f}{1175e6 / t / 1e9:>10.2f}')
    del stack, acc, w
    torch.cuda.empty_cache()


# --------------------------------------------------------------- Part B
def part_b(lib, dns, Smax, budget_list):
    print('=' * 100)
    print('Part B · 真机重叠：真实 SM90 GEMM（1D2D bf16）主流水 ∥ 增量消费者（侧流）')
    print('  控制变量：给 GEMM `set_num_sms(K)` 限核——K 越小留给消费者的 SM 槽位越多（隔离「占用」这个原因）')
    print('-' * 100)
    n_sms_b = torch.cuda.get_device_properties(0).multi_processor_count
    FULL = n_sms_b * 32
    stack = torch.empty(Smax, G, device=DEV, dtype=torch.bfloat16)
    gp = stack.data_ptr()
    w = torch.full((G,), 0.001, device=DEV, dtype=torch.float32)
    acc = torch.zeros(G, device=DEV, dtype=torch.bfloat16)
    wp, ap = w.data_ptr(), acc.data_ptr()
    side = torch.cuda.Stream()
    spp = ctypes.c_void_p(side.cuda_stream)
    cur = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    lr = 0.25

    def gemms_step(s):
        st = stack[s].view(E, M, N)
        for e, (a, b) in enumerate(dns[s]):
            deep_gemm.fp8_gemm_nt(a, b, st[e])

    def gemms(S):
        for s in range(S):
            gemms_step(s)

    def serial(S):
        def run():
            gemms(S)
            lib.c.woc_launch_all(gp, wp, G, S, lr, FULL, cur)
        return _bench(run, warm=3, iters=8)

    def overlap_inc(S, budget, which):
        def run():
            for s in range(S):
                ev = torch.cuda.Event()
                gemms_step(s)
                ev.record()
                side.wait_event(ev)
                if which == 'accbf16':
                    lib.c.woc_launch_acc(gp, ap, G, s, budget, spp)
                else:
                    lib.c.woc_launch_step(gp, wp, G, s, lr, budget, spp)
            ev2 = torch.cuda.Event()
            ev2.record(side)
            torch.cuda.current_stream().wait_event(ev2)
        return _bench(run, warm=3, iters=8)

    t_gemm_full = _bench(lambda: gemms(1), warm=4, iters=12)
    print(f'  单步 GEMM（默认 {n_sms_b} SM，{Smax} 步池子）：{t_gemm_full * 1e6:.1f} us/步')
    print(f'  {"gemm_sms":>9}{"gemm us/步":>12}{"SEQ8 us":>11}{"OVL8-INC-F":>12}{"OVL8-INC-B":>12}'
          f'{"SEQ/OVL-F":>11}{"SEQ/OVL-B":>11}')
    for k in (132, 96, 64, 48, 32):
        deep_gemm.set_num_sms(k)
        t_g = _bench(lambda: gemms(1), warm=4, iters=10)
        t_seq = serial(8)
        t_if = overlap_inc(8, FULL, 'fp32')
        t_ib = overlap_inc(8, FULL, 'accbf16')
        print(f'  {k:>9}{t_g * 1e6:>12.1f}{t_seq * 1e6:>11.1f}{t_if * 1e6:>12.1f}{t_ib * 1e6:>12.1f}'
              f'{t_seq / t_if:>11.2f}{t_seq / t_ib:>11.2f}')
    deep_gemm.set_num_sms(n_sms_b)

    print(f'  -- S=8 下扫增量消费者块数（GEMM 限 64 SM；看消费者要多快才能被藏住）--')
    deep_gemm.set_num_sms(64)
    t_seq64 = serial(8)
    print(f'  {"blocks":>8}{"INC-F us":>12}{"vs SEQ":>10}')
    for b in budget_list:
        t = overlap_inc(8, b, 'fp32')
        print(f'  {b:>8}{t * 1e6:>12.1f}{t_seq64 / t:>10.2f}')
    deep_gemm.set_num_sms(n_sms_b)
    del stack, w, acc
    torch.cuda.empty_cache()


# --------------------------------------------------------------- Part C
def part_c(lib, n_sms):
    print('=' * 100)
    print('Part C · 隔离控制：计算受限「伪 GEMM epilogue」生产者 ∥ 访存受限消费者')
    print('-' * 100)
    out = torch.empty(G, device=DEV, dtype=torch.bfloat16)
    op = out.data_ptr()
    stack = torch.randn(16, G, device=DEV, dtype=torch.bfloat16)
    gp = stack.data_ptr()
    w = torch.full((G,), 0.001, device=DEV, dtype=torch.float32)
    wp = w.data_ptr()
    side = torch.cuda.Stream()
    spp = ctypes.c_void_p(side.cuda_stream)
    cur = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    print(f'  {"flops":>7}{"prod-only us":>14}{"GB/s(prod)":>12}{"cons-only us":>14}{"ser us":>10}{"ovl us":>10}{"ovf":>7}')
    for flops in (256, 1024, 4096, 8192):
        t_p = lib.c.woc_gen(op, G, flops, n_sms, None, 15)
        t_c = lib.c.woc_fused_step(gp, wp, G, 0, 0.25, n_sms * 32, None, 15)

        def both():
            lib.c.woc_launch_gen(op, G, flops, n_sms, cur)
            lib.c.woc_launch_step(gp, wp, G, 0, 0.25, n_sms * 32, spp)
            ev2 = torch.cuda.Event(); ev2.record(side)
            torch.cuda.current_stream().wait_event(ev2)
        t_o = _bench(both, warm=3, iters=12)
        ovf = t_o / max(t_p, t_c)
        print(f'  {flops:>7}{t_p * 1e6:>14.1f}{G * 2 / t_p / 1e9:>12.2f}{t_c * 1e6:>14.1f}'
              f'{(t_p + t_c) * 1e6:>10.1f}{t_o * 1e6:>10.1f}{ovf:>7.2f}')
    print(f'  -- 决定性控制：固定 flops，扫生产者的**动态 smem**（模拟 DeepGEMM 的 ~227KB/SM 占用）--')
    print(f'  {"prod smem":>10}{"prod us":>10}{"ser us":>10}{"ovl us":>10}{"hidden%":>9}{"ovf":>7}')
    for sm in (0, 32768, 65536, 131072, 196608, 227 * 1024):
        t_p = lib.c.woc_gen_smem(op, G, 256, n_sms, sm, None, 15)
        t_c = lib.c.woc_fused_step(gp, wp, G, 0, 0.25, n_sms * 32, None, 15)

        def both2():
            lib.c.woc_launch_gen(op, G, 256, n_sms, cur)
            lib.c.woc_launch_step(gp, wp, G, 0, 0.25, n_sms * 32, spp)
            ev2 = torch.cuda.Event(); ev2.record(side)
            torch.cuda.current_stream().wait_event(ev2)
        # 注意：launch_gen 不带 smem；用带 smem 的单发版本
        t_o = None
        def run_smem():
            lib.c.woc_launch_gen_smem(op, G, 256, n_sms, sm, cur)   # producer（带 smem）
            lib.c.woc_launch_step(gp, wp, G, 0, 0.25, n_sms * 32, spp)
            ev2 = torch.cuda.Event(); ev2.record(side)
            torch.cuda.current_stream().wait_event(ev2)
        t_o = _bench(run_smem, warm=3, iters=10)
        hidden = (1.0 - (t_o - t_p) / t_c) * 100 if t_o > t_p else 100.0
        print(f'  {sm:>10}{t_p * 1e6:>10.1f}{(t_p + t_c) * 1e6:>10.1f}{t_o * 1e6:>10.1f}{hidden:>9.1f}{t_o / t_p:>7.2f}')
    del out, stack, w
    torch.cuda.empty_cache()


# --------------------------------------------------------------- Part D
def part_d():
    print('=' * 100)
    print('Part D · 增量协议的数值：bf16 累加器 S 步的累积舍入（vs fp64）')
    print('-' * 100)
    torch.manual_seed(0)
    S = 16
    g = torch.randn(S, G, device=DEV, dtype=torch.bfloat16)
    accbf = torch.zeros(G, device=DEV, dtype=torch.bfloat16)
    print(f'  {"S":>4}{"bf16-acc relerr":>18}{"fp32-sum relerr":>18}{"cos(bf16acc,ref)":>18}')
    run = torch.zeros(G, device=DEV, dtype=torch.float32)
    for s in range(S):
        accbf = (accbf.to(torch.float32) + g[s].to(torch.float32)).to(torch.bfloat16)
        run += g[s].to(torch.float32)
        if s + 1 in (1, 2, 4, 8, 16):
            ref = g[:s + 1].to(torch.float64).sum(0)   # 逐前缀参考（修正：不是全 S 的和）
            rr = ((accbf.to(torch.float64) - ref).norm() / ref.norm().clamp_min(1e-30)).item()
            rp = ((run.to(torch.float64) - ref).norm() / ref.norm().clamp_min(1e-30)).item()
            cos = torch.nn.functional.cosine_similarity(
                accbf.float().flatten(), ref.float().flatten(), dim=0).item()
            print(f'  {s + 1:>4}{rr:>18.3e}{rp:>18.3e}{cos:>18.7f}')
    del g
    torch.cuda.empty_cache()


def main():
    if not os.path.exists(LIB):
        print(f'缺少 {LIB}（先由 .sh 用 nvcc 编出）')
        return 1
    lib = Lib(LIB)
    n_sms = deep_gemm.get_num_sms()
    print(f'deep_gemm {deep_gemm.__version__}, device={torch.cuda.get_device_name(0)}, sms={n_sms}')
    host_ks = [align(KE, GRAN)] * E
    Smax = 16
    try:
        part_a(lib, n_sms * 32, n_sms)
    except Exception:
        print('Part A FAILED:\n' + traceback.format_exc())
    try:
        dns = build_operands(Smax, host_ks)
        part_b(lib, dns, Smax, (132, 264, 528, 1056, 2112, 4224))
    except Exception:
        print('Part B FAILED:\n' + traceback.format_exc())
    try:
        part_c(lib, n_sms)
    except Exception:
        print('Part C FAILED:\n' + traceback.format_exc())
    try:
        part_d()
    except Exception:
        print('Part D FAILED:\n' + traceback.format_exc())
    print('=' * 100)
    print('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
