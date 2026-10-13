"""P140-1 实验：把 MoE wgrad 的 S 步梯度累积「折进收缩维 K」——big-K 折叠能不能消掉每步累积税。

为什么做这篇（补的缺口）：
  第 145~149 篇把 MoE wgrad 慢的根因一路量化到底：
    真凶是**每专家一份 fp32 `[M,N]` 参数梯度**（第 145 篇，占 96% 流量），
    换 bf16 直写能快 2.27×（第 146 篇）但端到端封顶 ~1.55×（要物化 fp32 就得赔归约），
    「融合进优化器」把端到端拉到 1.67×（第 147 篇），
    「把归约藏进 GEMM」被 SM90 的 `232448 B/SM` 堵死（第 148 篇，双重否定），
    真 kernel EPI-B（bf16 累加器 RMW）虽能跑，但那多出来的 **235MB/步读彻底暴露（157%）**（第 149 篇）。
  第 149 篇结尾的结论是：**要再快只剩「减少字节」的路**；并点名两条：
    ① 更小的累加器/输出（FP8 累加器）；② **把「读一份 bf16 D」也省掉的分块归约**。

  本篇把第 ② 条走到能核对的数，得到一个**结构性**的答案：
  wgrad 的「每步累积」这件事本身，**只是因为我们把 S 个 micro-batch 当成 S 次独立 launch**。
  若把 S 步的激活**沿收缩维 K 拼接**（big-K 折叠），
      dW_e = Σ_s A_{e,s}^T B_{e,s} = [A_{e,1};…;A_{e,S}]^T [B_{e,1};…;B_{e,S}]，
  就**退化成一个 K=S·k_e 的单次 GEMM**：累加器全程住在 TMEM 里、只在最后一次 fresh 落盘，
  **跨 launch 的累积器读写（RMW）被彻底消掉**。这正是「省掉读一份 D」的路线。

要建立的结论（待实测）：
  ① big-K 与「逐 step k_grouped 累积」在数值上是**同一算子**（差异只是 fp8 量化噪声层）；
  ② big-K 把 S× 的 fp32 输出 RMW（每步读+写 `8B·E·M·N`）压成**一次 w 字节写**，
     端到端显著快于 REF（每步 fp32 累积）与 ALT-R（每步 fresh + 归约）；
  ③ 代价是**前提**：所有 S 步的激活必须**同时在手**（显存随 S 线性涨），
     以及 fp8 量化/累加顺序改变；
  ④ 大 K 下 big-K 自身**计算受限**（AI 高），不再受「每专家输出固定、k_e 小 → 访存受限」支配。

可跑性：全程 SM90 FP8 kernel（H100 可跑）。SM100 的 wgrad `tn` 为**仅分析**（见第 12 篇）。
本机 8 卡共享，本文在空闲 GPU 单进程测，报数注明。

输出重定向：harness/exp_wgrad_bigk.out.txt
"""
import os
import sys
import traceback

import torch

import deep_gemm

DG_DIR = os.environ.get('DEEPGEMM_DIR', os.path.expanduser('~/github/DeepGEMM'))
sys.path.insert(0, os.path.join(DG_DIR, 'tests'))

from deep_gemm.utils import align, per_token_cast_to_fp8, per_block_cast_to_fp8, per_channel_cast_to_fp8  # noqa: E402
from deep_gemm.utils import get_mk_alignment_for_contiguous_layout  # noqa: E402

GRAN = 128
DEV = 'cuda'

# wgrad dW2 形状（与第 135~149 篇一致）：[M=I=2048, N=H=7168]，每步 E=8 专家、每专家 k_e=512 token。
M, N, E, K_E = 2048, 7168, 8, 512
SUM_K = E * K_E                        # 单步收缩维（=T·topk=4096）
OUT_FP32 = E * M * N * 4               # 每步被读写的 fp32 输出（EE 份 [M,N]）
OUT_BF16 = E * M * N * 2


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-30)).item()


def _bench(fn, warm=3, iters=10):
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


def make_operands(S, seed=0):
    """每 (step, expert) 一块操作数，返回嵌套列表 operands[s][e] = (Xe[M,k_e], Ye[N,k_e])。"""
    torch.manual_seed(seed)
    ops = []
    for _ in range(S):
        step = []
        for _e in range(E):
            Xe = torch.randn(M, K_E, device=DEV)
            Ye = torch.randn(N, K_E, device=DEV)
            step.append((Xe, Ye))
        ops.append(step)
    return ops


# --- 逐 step 的 k_grouped 操作数（A_km [sum_k, M]、B_km [sum_k, N]，MN-major，逐 128 通道 SF）---
def quant_k_operand(X, host_ks):
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
    return data, (torch.cat(sfs).T if sfs else X.new_zeros((mn, 0)))


def make_k_grouped_step(step_ops, host_ks):
    """把一个 step 的专家块拼成 k_grouped 操作数（A^T [sum_k,M]、B^T [sum_k,N]）。"""
    A_km = torch.cat([Xe.t() for Xe, _ in step_ops], dim=0).contiguous()   # [sum_k, M]
    B_km = torch.cat([Ye.t() for _, Ye in step_ops], dim=0).contiguous()   # [sum_k, N]
    a = quant_k_operand(A_km, host_ks)
    b = quant_k_operand(B_km, host_ks)
    return a, b, (A_km, B_km)


def make_bigk_expert(ops, e):
    """把 S 个 step 的第 e 个专家块沿 K 拼接：X_cat[M, S·k_e]、Y_cat[N, S·k_e]（dense nt 方向）。"""
    X_cat = torch.cat([ops[s][e][0] for s in range(len(ops))], dim=1).contiguous()  # [M, S·k_e]
    Y_cat = torch.cat([ops[s][e][1] for s in range(len(ops))], dim=1).contiguous()  # [N, S·k_e]
    a = per_token_cast_to_fp8(X_cat, use_ue8m0=False, gran_k=GRAN)   # (fp8[M,K], sf)
    b = per_block_cast_to_fp8(Y_cat, use_ue8m0=False, gran_k=GRAN)   # (fp8[N,K], sf)
    return a, b, (X_cat, Y_cat)


# --------------------------------------------------------------------------- Part A
def part_a_correctness(S=4):
    print('=' * 96)
    print(f'Part A · big-K 折叠 == 逐 step 累积（同一 wgrad，S={S}）')
    print('-' * 96)
    deep_gemm.use_deterministic_algorithms(True)
    ops = make_operands(S, seed=7)
    host_ks = [align(K_E, GRAN)] * E

    # REF：逐 step k_grouped（fp32 强制累积），跨 step 累积进同一个 d[E,M,N]
    d_ref = torch.zeros(E, M, N, device=DEV, dtype=torch.float32)
    step_ops = []
    for s in range(S):
        a, b, raw = make_k_grouped_step(ops[s], host_ks)
        step_ops.append((a, b))
    for s in range(S):
        a, b = step_ops[s]
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            (a[0], a[1]), (b[0], b[1]), d_ref, host_ks, torch.tensor(host_ks, device=DEV, dtype=torch.int32),
            d_ref, recipe=(1, 1, GRAN))
    torch.cuda.synchronize()

    # BIG-K：逐专家一次 K=S·k_e 的稠密 1D2D（bf16 直写）
    bigk = []
    for e in range(E):
        a, b, _ = make_bigk_expert(ops, e)
        de = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
        deep_gemm.fp8_gemm_nt(a, b, de)
        bigk.append(de)
    torch.cuda.synchronize()

    diffs = [rel(d_ref[e], bigk[e]) for e in range(E)]
    # fp64 参考（用未量化 fp32 输入，看输出落点）
    for e in range(E):
        X = torch.cat([ops[s][e][0] for s in range(S)], dim=1).double()
        Y = torch.cat([ops[s][e][1] for s in range(S)], dim=1).double()
        ref64 = (X @ Y.t())
        print(f'  专家 {e}: reler(REF, BIG-K) = {diffs[e]:.3e}   '
              f'| BIG-K vs fp64 = {rel(bigk[e], ref64):.3e} | REF vs fp64 = {rel(d_ref[e], ref64):.3e}')
    print(f'  最大 REF-vs-BIG-K 相对误差 = {max(diffs):.3e}（~FP8 量化噪声 3.6e-2 层，证明是同一算子）')
    return max(diffs)


# --------------------------------------------------------------------------- Part B
def part_b_timing(S_LIST=(1, 2, 4, 8)):
    print('=' * 96)
    print('Part B · 端到端计时：REF(每步 k_grouped fp32/累积) vs ALT-R(每步 fresh+归约) vs BIG-K(折叠)')
    print('-' * 96)
    deep_gemm.use_deterministic_algorithms(True)
    host_ks = [align(K_E, GRAN)] * E
    layout = torch.tensor(host_ks, device=DEV, dtype=torch.int32)
    print(f'  {"S":>4}{"REF us":>10}{"ALT-R us":>11}{"BIG-K us":>11}{"bigk TFLOPS":>13}'
          f'{"REF/bigk":>10}{"ALT-R/bigk":>12}')
    for S in S_LIST:
        ops = make_operands(S, seed=1)
        fl = 2 * M * N * SUM_K * S

        d_ref = torch.zeros(E, M, N, device=DEV, dtype=torch.float32)
        ref_kops = [make_k_grouped_step(ops[s], host_ks)[:2] for s in range(S)]

        def ref_fn():
            d_ref.zero_()
            for s in range(S):
                a, b = ref_kops[s]
                deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
                    (a[0], a[1]), (b[0], b[1]), d_ref, host_ks, layout, d_ref, recipe=(1, 1, GRAN))

        # ALT-R：每 step、每专家 fresh bf16（K=k_e）落进 stack，最后一次 torch.sum 归约
        stack = torch.empty(S, E, M, N, device=DEV, dtype=torch.bfloat16)
        alt_ops = [[make_bigk_expert(ops[s:s + 1], e)[:2] for e in range(E)] for s in range(S)]

        def alt_fn():
            for s in range(S):
                for e in range(E):
                    a, b = alt_ops[s][e]
                    deep_gemm.fp8_gemm_nt(a, b, stack[s, e])
            torch.sum(stack, dim=0, dtype=torch.float32)

        bigk_ops = [make_bigk_expert(ops, e)[:2] for e in range(E)]

        def bigk_fn():
            for e in range(E):
                a, b = bigk_ops[e]
                deep_gemm.fp8_gemm_nt(a, b, stack[0, e])

        t_ref = _bench(ref_fn)
        t_alt = _bench(alt_fn)
        t_bigk = _bench(bigk_fn)
        print(f'  {S:>4}{t_ref * 1e6:>10.1f}{t_alt * 1e6:>11.1f}{t_bigk * 1e6:>11.1f}'
              f'{fl / t_bigk / 1e12:>13.0f}{t_ref / t_bigk:>10.2f}{t_alt / t_bigk:>12.2f}')
    print('  -> REF = 真实 wgrad 形态（1D1D k_grouped 强制 fp32 累积、每步 RMW E 份 [M,N]）；')
    print('     ALT-R = 每步 fresh bf16 + 最后一次归约；BIG-K = 沿 K 折叠 S 步、一次 fresh 落盘（无跨 launch 累积）')


# --------------------------------------------------------------------------- Part C
def part_c_bytes_and_memory(S_LIST=(1, 2, 4, 8)):
    print('=' * 96)
    print('Part C · 字节账 + 前提代价（激活驻留显存随 S 线性涨）')
    print('-' * 96)
    # 操作数字节（fp8）每步读一次：(M+N)*SUM_K*1B
    op_bytes = (M + N) * SUM_K * 1        # 单步
    print(f'  {"S":>4}{"REF 输出流量 MB":>18}{"BIG-K 输出流量 MB":>20}{"省下 MB":>12}'
          f'{"激活驻留 MB(fp8)":>18}{"AI_ref":>9}{"AI_bigk":>9}')
    for S in S_LIST:
        ref_out = 2 * OUT_FP32 * S                    # 每步读+写 fp32
        bigk_out = OUT_BF16                            # 一次 bf16 写
        saved = ref_out - bigk_out
        act = op_bytes * S                            # S 步操作数全驻留（fp8）
        fl = 2 * M * N * SUM_K * S
        ai_ref = fl / (op_bytes * S + ref_out + OUT_FP32)          # +最后物化一次 fp32? 这里输入已含
        ai_bigk = fl / (op_bytes * S + bigk_out)
        print(f'  {S:>4}{ref_out / 1e6:>18.1f}{bigk_out / 1e6:>20.1f}{saved / 1e6:>12.1f}'
              f'{act / 1e6:>18.1f}{ai_ref:>9.1f}{ai_bigk:>9.1f}')
    print(f'  注：AI=FLOP/DRAM 字节；H100 FP8 计算 roofline 拐点约 591 FLOP/B。')
    print('      REF 每步 RMW `8B·E·M·N`（fp32 读+写）；BIG-K 只在最后写一次 `2B·E·M·N`。')
    print('      BIG-K 的前提：S 步激活必须同时在手（fp8 版本显存 ≈ `(M+N)·SUM_K·S` 字节，'
          '未算 fp32 源张量）。')

    # 带宽锚：一次 fp32 [E,M,N] 的读+写 vs 一次 bf16 写
    fp32x = torch.empty(E * M * N, device=DEV, dtype=torch.float32)
    bf16x = torch.empty(E * M * N, device=DEV, dtype=torch.bfloat16)
    t_rw = _bench(lambda: fp32x.zero_())
    t_w = _bench(lambda: bf16x.zero_())
    print(f'  锚：写一份 fp32 D({OUT_FP32 / 1e6:.0f}MB) = {t_rw * 1e6:6.1f}us '
          f'({OUT_FP32 / t_rw / 1e12:.2f} TB/s)；写一份 bf16 D({OUT_BF16 / 1e6:.0f}MB) = '
          f'{t_w * 1e6:6.1f}us ({OUT_BF16 / t_w / 1e12:.2f} TB/s)')


def main():
    print(f'deep_gemm {deep_gemm.__version__}, device={torch.cuda.get_device_name(0)}, '
          f'sms={deep_gemm.get_num_sms()}')
    print(f'wgrad dW2 形状：[M={M}, N={N}]，每步 E={E} 专家、k_e={K_E}（单步收缩维 SUM_K={SUM_K}）；'
          f'每步 fp32 输出 {OUT_FP32 / 1e6:.0f}MB、bf16 {OUT_BF16 / 1e6:.0f}MB')
    for name, fn in (('Part A', part_a_correctness), ('Part B', part_b_timing),
                     ('Part C', part_c_bytes_and_memory)):
        try:
            fn()
        except Exception:
            print(f'{name} FAILED:\n' + traceback.format_exc())
    print('=' * 96)
    print('done')


if __name__ == '__main__':
    main()
