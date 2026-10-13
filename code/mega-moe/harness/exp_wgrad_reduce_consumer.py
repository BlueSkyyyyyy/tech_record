"""P137-1 实验：MoE wgrad 的「fp32 物化 / 归约税」能拉回多少 2.2×。

为什么做这篇（补的缺口）：
  第 146 篇（P136-1）落地了「wgrad 换 bf16 输出」：真实 SM90 GEMM 本身快 **2.27×**
  （`k_grouped` 1D1D fp32/累积 500.5us vs `E×` 稠密 1D2D bf16 220.8us），但**端到端封顶 ~1.55×**
  ——因为把每步 fresh 的 bf16 梯度再 `Σ` 成一份 fp32 参数梯度要付一笔**归约税**（S=8 时 838.6us）。
  它留下的**未答问题**：若让优化器**直接吃 bf16 梯度、不物化 fp32**，能拉回多少？本实验回答它。

三种协议（把 S 份 per-micro-batch 参数梯度交给优化器）：
  REF      : 每步真实 wgrad `k_grouped`（1D1D、D 必 fp32、**强制累积**，`gemm.hpp:384/403`）
  ALT-R    : 每步 `E×` 稠密 1D2D bf16 直写（`sm90_fp8_gemm_1d2d.hpp:81`），末次 `torch.sum` 成 fp32
  ALT-OPT  : 同上 bf16，但**融合进优化器**（`src/wgrad_reduce_consumer.cu` 的 `k_fused_sgd8`：
             读齐 S 份 bf16 就地求和 + 更新 fp32 master weight，**不落地 fp32 梯度**）

量纲：dW2 形状 `[E=8, M=2048, N=7168]`（DeepSeek 量级），`sum_k=4096`、每专家 `k_e=512`。
消费端统一用一个 SGD-lite `w ← w − lr·G`（AdamW 只多各自一份状态，对三协议是公共项）。

可跑性：全部 SM90 FP8 kernel（H100 可跑）；`wgrad_reduce_consumer.cu` 的消费侧内核全部架构无关。
SM100 的 wgrad `k_grouped_tn_contiguous` 同族 fp32 累积、仅分析。本机 8 卡共享，本文在**空闲
GPU1** 单进程测，报数注明。

输出重定向：harness/exp_wgrad_reduce_consumer.out.txt
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
LIB = os.path.join(HERE, 'wgrad_reduce_consumer.so')


# --------------------------------------------------------------------------- ctypes
class Lib:
    def __init__(self, path):
        self.c = ctypes.CDLL(path)
        c = self.c
        c.wrc_malloc_bf16.argtypes = [ctypes.c_long]; c.wrc_malloc_bf16.restype = ctypes.c_void_p
        c.wrc_malloc_f32.argtypes = [ctypes.c_long]; c.wrc_malloc_f32.restype = ctypes.c_void_p
        c.wrc_free.argtypes = [ctypes.c_void_p]
        c.wrc_read.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_int]; c.wrc_read.restype = ctypes.c_double
        c.wrc_reduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_int]
        c.wrc_reduce.restype = ctypes.c_double
        c.wrc_atomic.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_int]
        c.wrc_atomic.restype = ctypes.c_double
        c.wrc_fused.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_int,
                                ctypes.c_double, ctypes.c_int]
        c.wrc_fused.restype = ctypes.c_double
        c.wrc_selftest.argtypes = [ctypes.POINTER(ctypes.c_double)]


def _bench(fn, warm=10, iters=30):
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


# --------------------------------------------------------------------------- operands (同 P136)
def quant_kg(X, host_ks):
    tk, mn = X.shape
    data = torch.zeros(tk * mn, dtype=torch.float8_e4m3fn, device=X.device)
    sfs = []
    off = 0
    for hk in host_ks:
        xg = X[off:off + hk].contiguous()
        xf, sf = per_channel_cast_to_fp8(xg, use_ue8m0=False, gran_k=GRAN)
        data[off * mn:(off + hk) * mn] = xf.T.reshape(-1)
        sfs.append(sf)
        off += hk
    sf_cat = torch.cat(sfs).T if sfs else X.new_zeros((mn, 0))
    return data, sf_cat


def kg_operands(Ag, Bg, host_ks):
    ad, asf = quant_kg(Ag, host_ks)
    bd, bsf = quant_kg(Bg, host_ks)
    return (ad, asf), (bd, bsf)


def dense_operands(Ag, Bg, host_ks):
    ops = []
    off = 0
    for k in host_ks:
        A = Ag[off:off + k].T.contiguous()
        B = Bg[off:off + k].T.contiguous()
        a = per_token_cast_to_fp8(A, use_ue8m0=False, gran_k=GRAN)
        b = per_block_cast_to_fp8(B, use_ue8m0=False, gran_k=GRAN)
        ops.append((a, b))
        off += k
    return ops


# --------------------------------------------------------------------------- parts
def part_a_kernels(lib):
    print('=' * 96)
    print('Part A · 消费侧内核的正确性 + 与聚合/atomic 变体的带宽对照')
    print('-' * 96)
    err = (ctypes.c_double * 2)()
    lib.c.wrc_selftest(err)
    print(f'  正确性（GPU vs CPU double）：reduce max|err|={err[0]:.3e}  fused_sgd max|err|={err[1]:.3e}'
          f'  -> {"OK" if max(err[0], err[1]) < 1e-2 else "FAIL"}')

    E, M, N = 8, 2048, 7168
    G = E * M * N
    S_list = (1, 2, 4, 8, 16)
    Smax = max(S_list)
    torch.cuda.empty_cache()
    # 直接用 torch 张量，把 device 指针交给自定义核（不跨 host/device memmove）
    stack = torch.randn(Smax, G, device=DEV, dtype=torch.bfloat16)
    gp = stack.data_ptr()
    out_fp32 = torch.empty(G, device=DEV, dtype=torch.float32)
    w_buf = torch.full((G,), 0.001, device=DEV, dtype=torch.float32)

    t_read = lib.c.wrc_read(gp, G, 20)
    print(f'  纯读一份 bf16 ({G * 2 / 1e6:.0f} MB)：{t_read * 1e6:.1f} us  '
          f'({G * 2 / t_read / 1e9:.2f} GB/s，HBM 峰值参照)')
    print(f'  {"核/协议":<28}{"时(us)":>10}{"读(MB)":>10}{"写(MB)":>10}{"总(GB/s)":>12}')
    for S in S_list:
        t = lib.c.wrc_reduce(gp, out_fp32.data_ptr(), G, S, 10)
        rb, wb = S * G * 2, G * 4
        print(f'  {"custom reduce8 S=%d" % S:<28}{t * 1e6:>10.1f}{rb / 1e6:>10.0f}{wb / 1e6:>10.0f}'
              f'{(rb + wb) / t / 1e9:>12.2f}')
        t = lib.c.wrc_fused(gp, w_buf.data_ptr(), G, S, 0.25, 10)
        rb, wb = S * G * 2 + G * 4, G * 4
        print(f'  {"custom fused_sgd S=%d" % S:<28}{t * 1e6:>10.1f}{rb / 1e6:>10.0f}{wb / 1e6:>10.0f}'
              f'{(rb + wb) / t / 1e9:>12.2f}')
        if S == 1:   # atomic 走 RMW，极慢；只留一个 S 说明趋势
            t = lib.c.wrc_atomic(gp, out_fp32.data_ptr(), G, S, 3)
            rb, wb = S * G * 2, S * G * 4
            print(f'  {"custom atomic S=%d" % S:<28}{t * 1e6:>10.1f}{rb / 1e6:>10.0f}{wb / 1e6:>10.0f}'
                  f'{(rb + wb) / t / 1e9:>12.2f}')

    # 与 torch.sum 对照（ALT-R 用的就是它）
    print(f'  {"torch.sum(dim=0,fp32) 对照":<28}')
    for S in S_list:
        t = _bench(lambda: torch.sum(stack[:S], dim=0, dtype=torch.float32, out=out_fp32))
        rb, wb = S * G * 2, G * 4
        print(f'  {"torch.sum S=%d" % S:<28}{t * 1e6:>10.1f}{rb / 1e6:>10.0f}{wb / 1e6:>10.0f}'
              f'{(rb + wb) / t / 1e9:>12.2f}')
    del stack, out_fp32, w_buf
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()


def part_b_protocols(lib):
    print('=' * 96)
    print('Part B · 三协议端到端（真实 GEMM + 消费）：REF / ALT-R / ALT-OPT')
    print('-' * 96)
    M, N, E, k_e = 2048, 7168, 8, 512
    host_ks = [align(k_e, GRAN)] * E
    sum_k = sum(host_ks)
    G = E * M * N
    S_LIST = (1, 2, 4, 8, 16)
    Smax = max(S_LIST)
    torch.manual_seed(2)
    layout = torch.tensor(host_ks, device=DEV, dtype=torch.int32)

    kgs, dns = [], []
    for _ in range(Smax):
        Ag = torch.randn(sum_k, M, device=DEV)
        Bg = torch.randn(sum_k, N, device=DEV)
        kgs.append(kg_operands(Ag, Bg, host_ks))
        dns.append(dense_operands(Ag, Bg, host_ks))
        del Ag, Bg
    torch.cuda.empty_cache()

    # 真实产物：REF 累积 / ALT bf16 每步
    Gref = torch.zeros(E, M, N, device=DEV, dtype=torch.float32)
    stack = torch.empty(Smax, G, device=DEV, dtype=torch.bfloat16)
    for s in range(Smax):
        st = stack[s].view(E, M, N)
        for e, (a, b) in enumerate(dns[s]):
            deep_gemm.fp8_gemm_nt(a, b, st[e])
    torch.cuda.synchronize()

    fl = 2 * M * N * sum_k

    def ref_step():
        (ad, asf), (bd, bsf) = kgs[0]
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            (ad, asf), (bd, bsf), Gref, host_ks, layout, Gref, recipe=(1, 1, GRAN))

    def alt_step():
        for e, (a, b) in enumerate(dns[0]):
            deep_gemm.fp8_gemm_nt(a, b, stack[0].view(E, M, N)[e])

    t_ref = _bench(ref_step)
    t_alt = _bench(alt_step)
    print(f'  单步 GEMM（墙体，各 30 次）：REF `k_grouped` fp32/acc {t_ref * 1e6:.1f}us '
          f'({fl / t_ref / 1e12:.0f} TFLOPS)、ALT E×稠密 1D2D bf16 {t_alt * 1e6:.1f}us '
          f'({fl / t_alt / 1e12:.0f} TFLOPS)、纯 GEMM 比 {t_ref / t_alt:.2f}×')

    # 消费端（自定义核直接读 torch 张量的 device 指针，不跨 host/device 拷贝）
    gp = stack.data_ptr()
    w_master = torch.full((G,), 0.001, device=DEV, dtype=torch.float32)
    lr = 0.25

    # 消费端时间（一次，覆盖 S 份梯度）
    def opt_ref():  # 优化器读 fp32 梯度 + 读/写 master
        w_master.add_(Gref.reshape(-1), alpha=-lr)

    opt_ref_t = _bench(opt_ref)

    print(f'  消费端单次：REF opt(add_ fp32) {opt_ref_t * 1e6:.1f}us')
    print(f'  {"S":>4}{"torch.sum us":>14}{"fused us":>12}{"REF e2e us":>13}'
          f'{"ALT-R e2e":>12}{"ALT-OPT e2e":>13}{"REF/ALT-R":>11}{"REF/ALT-OPT":>13}')
    out = {}
    for S in S_LIST:
        t_sum = _bench(lambda: torch.sum(stack[:S], dim=0, dtype=torch.float32, out=out_fp32))
        t_fused = lib.c.wrc_fused(gp, w_master.data_ptr(), G, S, lr, 10)
        e_ref = t_ref * S + opt_ref_t
        e_altr = t_alt * S + t_sum + opt_ref_t
        e_altopt = t_alt * S + t_fused
        print(f'  {S:>4}{t_sum * 1e6:>14.1f}{t_fused * 1e6:>12.1f}{e_ref * 1e6:>13.1f}'
              f'{e_altr * 1e6:>12.1f}{e_altopt * 1e6:>13.1f}{e_ref / e_altr:>11.2f}'
              f'{e_ref / e_altopt:>13.2f}')
        out[S] = (e_ref, e_altr, e_altopt)
    gemm_ratio = t_ref / t_alt
    print(f'  -> 纯 GEMM 比（上界）= {gemm_ratio:.2f}×；'
          f'ALT-R 把上界兑现成 {out[8][0] / out[8][1]:.2f}×（S=8），'
          f'ALT-OPT 兑现成 {out[8][0] / out[8][2]:.2f}×')
    print(f'  -> 「物化税」= reduce 写 fp32 + 优化器再读回；融合后省掉这两步。')


def main():
    if not os.path.exists(LIB):
        print(f'缺少 {LIB}（先由 .sh 用 nvcc 编出）')
        return 1
    lib = Lib(LIB)
    print(f'deep_gemm {deep_gemm.__version__}, device={torch.cuda.get_device_name(0)}, '
          f'sms={deep_gemm.get_num_sms()}')
    # out_fp32 需要在 part_a / part_b 可见
    global out_fp32
    out_fp32 = torch.empty(8 * 2048 * 7168, device=DEV, dtype=torch.float32)
    for name, fn in (('Part A', part_a_kernels), ('Part B', part_b_protocols)):
        try:
            fn(lib)
        except Exception:
            print(f'{name} FAILED:\n' + traceback.format_exc())
    print('=' * 96)
    print('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
