"""P141-1 实验：MoE wgrad 的「FP8 累加器」路线——把每步累积器 RMW 从 8B 压到 2B 的
   **误差-字节边界**，与 big-K 折叠（第 150 篇）对比。

为什么做这篇（补的缺口）：
  第 150 篇把 wgrad 的 S 步累积「折进收缩维 K」（big-K），给出结构性正面答案：跨 launch 的
  累积器 RMW 归零、端到端 4.76×。但它结尾留了一条**没做的路线**（原话）：
    ① 更小的累加器/输出（**FP8 累加器**）：把每步 RMW 的 8B 压到 2B。
  并预判「FP8 只有 2~3 位有效数字、动态范围窄，跨 S 步随机符号求和的误差预计不可用」。

  本篇把这条路线走到能核对的数，得到它对 big-K 的**误差-字节边界**：
    - **字节侧**（src/fp8_accum_rmw.cu，真实内核）：fp32 累积器 8B/元素/步 → bf16 4B → fp8 2B。
    - **误差侧**（本文件 Part A/B）：FP8 累加器的相对误差**随 S 近似线性增长**，在 S≈2~4 就
      超过 FP8 输入量化噪声（~3.6e-2）——坐实第 150 篇的预判。
    - **结论**：路线 ① 在误差侧被封死；只有在「激活无法同时驻留、必须流式累积」时才谈得上，
      而那时误差已不可用。big-K 在**字节和误差两侧同时占优**（前提是 S 步激活能驻留）。

三个 Part：
  A. 隔离累加器精度：逐 step 用 fp32（非 TF32）matmul 得**精确 partial**，按不同累积协议
     （fp32 / bf16 / fp8-逐张量 / fp8-逐 128×128 块 / fp8-UE8M0 块）累积 S 步，对 fp64 参考。
     误差只来自累积器舍入，与 GEMM/输入量化无关。S∈{1,2,4,8,16,32}。
  B. 真实 SM90 kernel：每步用 `k_grouped_fp8_gemm_nt_contiguous`（c=0 → fresh fp32 partial），
     在真实 partial 上跑累积协议，报 vs fp64 与 FP8 输入噪声地板。E=8、S∈{1,2,4,8}。
  C. 字节账 + 每步 RMW 模型时间（用 src/fp8_accum_rmw.cu 实测到的带宽）。

可跑性：全程 SM90 FP8 kernel（H100 可跑）。SM100 wgrad `tn` 为仅分析（见第 12 篇）。
本机 8 卡共享，本文在空闲 GPU 单进程测，报数注明。

输出重定向：harness/exp_wgrad_fp8_accum.out.txt
"""
import os
import sys
import traceback

import torch

import deep_gemm

DG_DIR = os.environ.get('DEEPGEMM_DIR', os.path.expanduser('~/github/DeepGEMM'))
sys.path.insert(0, os.path.join(DG_DIR, 'tests'))

from deep_gemm.utils import align, per_channel_cast_to_fp8  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

GRAN = 128
DEV = 'cuda'
FP8_MAX = 448.0

# wgrad dW2 形状（与第 135~150 篇一致）
M, N, E, K_E = 2048, 7168, 8, 512
SUM_K = E * K_E
OUT_FP32 = E * M * N * 4
OUT_BF16 = E * M * N * 2
OUT_FP8 = E * M * N * 1


def rel(a, b):
    a, b = a.float().double(), b.float().double()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-30)).item()


def cos(a, b):
    a, b = a.float().flatten().double(), b.float().flatten().double()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


# ------------------------------------------------------------------ fp8 参考量化器（逐张量 / 逐块）
def q_fp8_tensor(x):
    """逐张量 amax 缩放的 E4M3 量化-反量化（返回 fp32）。"""
    amax = x.abs().amax()
    scale = torch.clamp(amax / FP8_MAX, min=1e-12)
    q = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.float() * scale


def _q_fp8_block(x, BM, BN, ue8m0):
    lead = x.shape[:-2]
    Ms, Ns = x.shape[-2], x.shape[-1]
    xf = x.reshape(-1, Ms, Ns)
    B = xf.shape[0]
    xb = xf.reshape(B, Ms // BM, BM, Ns // BN, BN).permute(0, 1, 3, 2, 4)   # [B,mb,nb,BM,BN]
    amax = xb.abs().amax(dim=(3, 4), keepdim=True)
    if ue8m0:
        # 2 的幂 scale（>= amax/448 的最小 2 的幂）——DeepGEMM 的 UE8M0 SF
        t = torch.clamp(amax / FP8_MAX, min=1e-12)
        scale = torch.pow(2.0, torch.ceil(torch.log2(t)))
    else:
        scale = torch.clamp(amax / FP8_MAX, min=1e-12)
    q = (xb / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    dq = q.float() * scale
    out = dq.permute(0, 1, 3, 2, 4).reshape(B, Ms, Ns)
    return out.reshape(*lead, Ms, Ns)


def q_fp8_block(x, BM=128, BN=128):
    return _q_fp8_block(x, BM, BN, ue8m0=False)


def q_fp8_block_ue8m0(x, BM=128, BN=128):
    return _q_fp8_block(x, BM, BN, ue8m0=True)


# ------------------------------------------------------------------ 累积协议（fp32 累加器语义，但存储被量化）
def acc_fp32(acc, dp):
    return acc + dp


def acc_bf16(acc, dp):
    return (acc + dp).to(torch.bfloat16).float()


def acc_fp8_tensor(acc, dp):
    return q_fp8_tensor(acc + dp)


def acc_fp8_block(acc, dp):
    return q_fp8_block(acc + dp)


def acc_fp8_block_ue8m0(acc, dp):
    return q_fp8_block_ue8m0(acc + dp)


PROTOCOLS = {
    'fp32': acc_fp32,
    'bf16': acc_bf16,
    'fp8/T': acc_fp8_tensor,
    'fp8/B128': acc_fp8_block,
    'fp8/B128·ue8m0': acc_fp8_block_ue8m0,
}


# --------------------------------------------------------------------------- Part A
def part_a_isolated(S_LIST=(1, 2, 4, 8, 16, 32)):
    print('=' * 100)
    print('Part A · 隔离累加器精度：fp32 精确 partial + 不同累积协议 vs fp64（只含累积器舍入）')
    print('-' * 100)
    print(f'  形状：每 step 一个 partial [M={M}, N={N}]（E=1，精确 fp32 matmul，allow_tf32=False）')
    print(f'  {"S":>4}' + ''.join(f'{k:>16}' for k in PROTOCOLS))
    results = {k: [] for k in PROTOCOLS}
    for S in S_LIST:
        torch.manual_seed(1234)
        accs = {k: torch.zeros(M, N, device=DEV, dtype=torch.float32) for k in PROTOCOLS}
        ref = torch.zeros(M, N, device=DEV, dtype=torch.float64)
        for s in range(S):                     # 逐 step 现产操作数（不驻留）
            torch.manual_seed(1000 + s)
            X = torch.randn(M, K_E, device=DEV)
            Y = torch.randn(N, K_E, device=DEV)
            dp = X @ Y.t()                     # fp32 精确 partial（本步梯度）
            ref += dp.double()
            for k, fn in PROTOCOLS.items():
                accs[k] = fn(accs[k], dp)
        row = f'  {S:>4}'
        for k in PROTOCOLS:
            e = rel(accs[k], ref)
            results[k].append(e)
            row += f'{e:>16.3e}'
        print(row)
    print(f'  注：FP8 输入量化噪声地板 ≈ 3.6e-2（第 146/150 篇）。逐 step partial 越密，'
          f'累积器被量化次数 = S。')
    print(f'  cos(acc, ref)：', end='')
    return results


def part_a_cos(S=16):
    torch.manual_seed(1234)
    accs = {k: torch.zeros(M, N, device=DEV, dtype=torch.float32) for k in PROTOCOLS}
    ref = torch.zeros(M, N, device=DEV, dtype=torch.float64)
    for s in range(S):
        torch.manual_seed(1000 + s)
        X = torch.randn(M, K_E, device=DEV)
        Y = torch.randn(N, K_E, device=DEV)
        dp = X @ Y.t()
        ref += dp.double()
        for k, fn in PROTOCOLS.items():
            accs[k] = fn(accs[k], dp)
    out = ' '.join(f'{k}={cos(accs[k], ref):.6f}' for k in PROTOCOLS)
    print(f'    (S={S}) ' + out)


# --------------------------------------------------------------------------- Part B：真实 kernel 操作数
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
    A_km = torch.cat([Xe.t() for Xe, _ in step_ops], dim=0).contiguous()
    B_km = torch.cat([Ye.t() for _, Ye in step_ops], dim=0).contiguous()
    return quant_k_operand(A_km, host_ks), quant_k_operand(B_km, host_ks)


def part_b_real_kernel(S_LIST=(1, 2, 4, 8)):
    print('=' * 100)
    print('Part B · 真实 SM90 kernel：每步 k_grouped(c=0) 得 fresh fp32 partial，累积协议 vs fp64')
    print('-' * 100)
    deep_gemm.use_deterministic_algorithms(True)
    host_ks = [align(K_E, GRAN)] * E
    layout = torch.tensor(host_ks, device=DEV, dtype=torch.int32)
    protos = ['fp32', 'bf16', 'fp8/B128', 'fp8/B128·ue8m0']
    print(f'  {"S":>4}' + ''.join(f'{k:>18}' for k in protos) + f'{"FP8输入噪声":>14}')
    for S in S_LIST:
        torch.manual_seed(7)
        accs = {k: torch.zeros(E, M, N, device=DEV, dtype=torch.float32) for k in protos}
        ref = torch.zeros(E, M, N, device=DEV, dtype=torch.float64)
        noise = 0.0
        for s in range(S):
            step_ops = []
            for _e in range(E):
                Xe = torch.randn(M, K_E, device=DEV)
                Ye = torch.randn(N, K_E, device=DEV)
                step_ops.append((Xe, Ye))
            (a0, a1), (b0, b1) = make_k_grouped_step(step_ops, host_ks)
            d = torch.empty(E, M, N, device=DEV, dtype=torch.float32)
            c0 = torch.zeros_like(d)
            deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
                (a0, a1), (b0, b1), d, host_ks, layout, c0, recipe=(1, 1, GRAN))
            torch.cuda.synchronize()
            if s == 0:                          # FP8 输入噪声（真实 partial vs fp64 精确）
                for e in range(E):
                    Xe, Ye = step_ops[e]
                    ref64 = Xe.double() @ Ye.double().t()
                    noise = max(noise, rel(d[e], ref64))
            ref += d.double()
            for k in protos:
                accs[k] = PROTOCOLS[k](accs[k], d)
        row = f'  {S:>4}'
        for k in protos:
            row += f'{rel(accs[k], ref):>18.3e}'
        row += f'{noise:>14.3e}'
        print(row)
    print(f'  注：真实 kernel 的 partial 本身带 FP8 输入量化噪声（末列），累积协议在其上再叠舍入。')


# --------------------------------------------------------------------------- Part C
def part_c_bytes(S_LIST=(1, 2, 4, 8, 16)):
    print('=' * 100)
    print('Part C · 字节账（真实每步 RMW 时间见 src/fp8_accum_rmw.cu 的带宽报告）')
    print('-' * 100)
    nelem = E * M * N
    print(f'  累积器元素数 = E*M*N = {nelem/1e6:.1f}M（dW2 [E=8, M=2048, N=7168]）')
    print(f'  {"S":>4}{"FP32 RMW MB":>14}{"BF16 RMW MB":>14}{"FP8 RMW MB":>14}'
          f'{"FP8省vsFP32":>14}{"bigK MB":>10}{"bigK省vsFP32":>15}')
    for S in S_LIST:
        mb32 = S * 2 * nelem * 4 / 1e6
        mb16 = S * 2 * nelem * 2 / 1e6
        mb8 = S * 2 * nelem * 1 / 1e6
        bigk = 1 * nelem * 2 / 1e6               # big-K 只在最后一次写一份 bf16（无 RMW）
        print(f'  {S:>4}{mb32:>14.1f}{mb16:>14.1f}{mb8:>14.1f}'
              f'{mb32 / mb8:>13.1f}x{bigk:>10.1f}{mb32 / bigk:>14.1f}x')
    print('  注：RMW 字节 = S·2·size·E·M·N（每步读+写累积器）；big-K = 一次 bf16 写（无跨 launch RMW）。')
    print('      FP8 累积器相对 fp32 省 4×、相对 bf16 省 2×；但 big-K 把 RMW 直接归零（省 S×）。')


def main():
    print(f'deep_gemm {deep_gemm.__version__}, device={torch.cuda.get_device_name(0)}, '
          f'sms={deep_gemm.get_num_sms()}')
    print(f'wgrad dW2 [M={M}, N={N}]，E={E}，k_e={K_E}，SUM_K={SUM_K}')
    for name, fn in (('Part A', part_a_isolated), ('Part A-cos', part_a_cos),
                     ('Part B', part_b_real_kernel), ('Part C', part_c_bytes)):
        try:
            fn()
        except Exception:
            print(f'{name} FAILED:\n' + traceback.format_exc())
    print('=' * 100)
    print('done')


if __name__ == '__main__':
    main()
