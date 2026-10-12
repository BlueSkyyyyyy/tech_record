"""P2-1 实验：`transform_weights_for_mega_moe` 三种重排的纯 PyTorch 核对。

Mega MoE 的权重不是直接喂进 kernel 的：host 侧要先做两件事——
  1) L1 权重按 8 粒度交错 gate/up（`_interleave_weights`）；
  2) FP8/FP4 权重的 UE8M0 scale 按 UTCCP 的 4x32 pattern 转置
     （`_transpose_sf_for_utccp`）。
本脚本在 H100（sm90）上用纯 PyTorch 复现这两个变换，并与 DeepGEMM 内部实现、
以及 kernel 内 `transform_sf_token_idx`（`sm100_fp8_fp4_mega_moe.cuh:145-149`）逐元素核对。
不涉及 SM100 kernel，仅验证**棋盘式重排的语义**。

来源核对（`~/github/DeepGEMM`，2026-09-30 公开版 commit `057ca59`）：
  - `deep_gemm/mega/__init__.py:112-164`（`_interleave_weights` / `_transpose_sf_for_utccp` /
    `transform_weights_for_mega_moe`）
  - `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:145-149`（UTCCP 4x32 索引映射）
  - `...cuh:1066-1134`（L1 epilogue 用 gran-8 交错权重做 in-place SwiGLU）
"""
import os
import sys

import torch


def align(x, y):
    return (x + y - 1) // y * y


# ---- 参考实现（与 deep_gemm/mega/__init__.py:112-143 语义一致）----
def interleave_weights_ref(t, gran=8):
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...]
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    return torch.stack([gate, up], dim=2).reshape(g, n, *rest)


def transpose_sf_for_utccp_ref(sf):
    # [g, mn, packed_k] -> 每个 128 行组内做 (4,32)->(32,4) 转置
    g, mn, packed_k = sf.shape
    assert mn % 128 == 0
    return (sf.reshape(g, -1, 4, 32, packed_k)
              .transpose(2, 3)
              .reshape(g, mn, packed_k))


def kernel_transform_sf_token_idx(token_idx_in_expert, block_m):
    # sm100_fp8_fp4_mega_moe.cuh:145-149
    idx = token_idx_in_expert % block_m
    return (token_idx_in_expert // block_m) * align(block_m, 128) \
        + (idx & ~127) + (idx & 31) * 4 + ((idx >> 5) & 3)


def main():
    torch.manual_seed(0)
    ok = True

    # ---------- 1) L1 gate/up 交错：行号映射 ----------
    g, n = 2, 32  # half = 16 -> 两个 gran=8 组，能看出交错
    # 行标签：前半 gate_j，后半 up_j
    labels = [f'G{j}' for j in range(n // 2)] + [f'U{j}' for j in range(n // 2)]
    # 整数编码：gate_j = j, up_j = 100+j
    code = torch.tensor([[j for j in range(n // 2)] + [100 + j for j in range(n // 2)]
                         for _ in range(g)], dtype=torch.float)
    inter = interleave_weights_ref(code)
    print('[interleave] 输入行序 (前 8 = gate, 后 8 = up):')
    print('             ', [labels[i] for i in range(n)])
    rowmap = []
    for v in inter[0].tolist():
        rowmap.append(('G' if v < 100 else 'U') + str(int(v % 100)))
    print('[interleave] 输出行序:')
    print('             ', rowmap)
    expected = [x for b in range(n // 2 // 8) for x in
                [f'G{8*b+i}' for i in range(8)] + [f'U{8*b+i}' for i in range(8)]]
    print(f'[interleave] 与期望 [G0..7,U0..7,G8..,U8..] 一致: {rowmap == expected}')
    ok &= rowmap == expected

    # ---------- 2) 交错后 SwiGLU 语义等价 ----------
    E, I, H = 3, 64, 32
    W1 = torch.randn(E, 2 * I, H)          # [gate | up]
    W1i = interleave_weights_ref(W1)       # [E, 2I, H]
    x = torch.randn(5, H)
    y_orig = x @ W1[0].t()                 # [5, 2I]
    y_int = x @ W1i[0].t()                 # [5, 2I]
    gate_o, up_o = y_orig[:, :I], y_orig[:, I:]
    swiglu_orig = torch.nn.functional.silu(gate_o) * up_o

    # 交错布局下：组 b（8 个）后的 8 行是配对的 up；即 [..., G8b..G8b+7, U8b..U8b+7, ...]
    gate_i = torch.empty(5, I)
    up_i = torch.empty(5, I)
    for b in range(I // 8):
        gate_i[:, 8 * b:8 * b + 8] = y_int[:, 16 * b:16 * b + 8]
        up_i[:, 8 * b:8 * b + 8] = y_int[:, 16 * b + 8:16 * b + 16]
    swiglu_int = torch.nn.functional.silu(gate_i) * up_i
    err = (swiglu_orig - swiglu_int).abs().max().item()
    print(f'\n[SwiGLU] 交错前后 silu(gate)*up 最大绝对误差: {err:.3e}')
    ok &= err < 1e-5

    # ---------- 3) SF 的 UTCCP 4x32 转置 ----------
    sf = torch.arange(256, dtype=torch.int32).reshape(1, 256, 1)
    sf_t = transpose_sf_for_utccp_ref(sf)
    # 组内映射：原行 r -> (r&~127) + (r&31)*4 + ((r>>5)&3)
    bad = 0
    for r in range(256):
        want = (r & ~127) + (r & 31) * 4 + ((r >> 5) & 3)
        got = int(sf_t[0, :, 0].tolist().index(r))
        bad += (want != got)
    print(f'\n[utccp] 256 行里与 kernel `transform_sf_token_idx` 不符的行数: {bad}')
    ok &= bad == 0
    # 打印前 16 行映射便于肉眼核对
    print('[utccp] 原行 -> 新行 (前 16):')
    print('       ', [(r, (r & ~127) + (r & 31) * 4 + ((r >> 5) & 3)) for r in range(16)])

    # ---------- 4) 与 DeepGEMM 内部实现交叉核对（若可 import）----------
    try:
        from deep_gemm.mega import _interleave_weights, _transpose_sf_for_utccp
        a = torch.randn(4, 48, 16)
        b1 = interleave_weights_ref(a)
        b2 = _interleave_weights(a)
        e1 = (b1 - b2).abs().max().item()
        s = torch.randint(0, 1000, (2, 256, 3), dtype=torch.int32)
        c1 = transpose_sf_for_utccp_ref(s)
        c2 = _transpose_sf_for_utccp(s)
        e2 = (c1 - c2).abs().max().item()
        print(f'\n[cross-check] vs deep_gemm.mega._interleave_weights 最大差: {e1}')
        print(f'[cross-check] vs deep_gemm.mega._transpose_sf_for_utccp 最大差: {e2}')
        ok &= (e1 == 0) and (e2 == 0)
    except Exception as ex:  # pragma: no cover
        print(f'\n[cross-check] 跳过（deep_gemm 不可用）: {ex}')

    print(f'\nALL_OK = {ok}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
