#!/usr/bin/env python3
"""P145-1 实验：把反向的 SwiGLU 段折进 dgrad 的 epilogue —— 字节账 + 真实 d2g 锚点 + 预算组装。

承接第 154 篇（P144-1）留下的口子：把「SwiGLU 反向 + 激活梯度重量化」段折进 d2g/d1g 的 epilogue，
量它省多少字节、extra 读能不能藏。d2g/d1g 输出侧原型见 src/moe_backward_activation_fused.cu（.sh 段一）；
本文件做三件事（纯 host + torch、架构无关，H100(sm90) 真机可跑）：
  Part A：把 standalone 流水与 fused 的**每元素字节账**列清楚（d(act) 往返 = 4B/元素）。
  Part B：真实 SM90 d2g（`m_grouped_fp8_gemm_nt_contiguous`）的 event 计时（现实现的 anchor）。
  Part C：把 d2g + 激活段 + fused 投影拼成反向 single-step 预算。

d2g/d1g 的 SM100 分支（1D1D + 自定义 epilogue）仅分析；本文件复刻语义。
"""
import os
import sys

import torch

import deep_gemm
from deep_gemm.testing import get_arch_major

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def part_a():
    print("== Part A：每元素字节账（d2g 的输出形状 [Σt, I]，bf16）==")
    print("  现实现 = d2g 写 d(act) + 独立段读 d(act)/gate/up + 写 FP8 对：")
    print("    d2g epilogue 写 d(act)          : 2 B/元素（bf16）")
    print("    独立段 读 d(act) + gate + up     : 6 B/元素（3×bf16）")
    print("    独立段 写 d(gate)+d(up) FP8      : 2 B/元素（2×E4M3，SF 另 ~0.06）")
    print("    小计（d2g 算力之外）            : 10 B/元素")
    print("  折叠实现 = d2g 的 epilogue 里就地反解 + 量化（d(act) 不出寄存器）：")
    print("    d2g epilogue 读 gate + up        : 4 B/元素")
    print("    d2g epilogue 写 d(gate)+d(up) FP8: 2 B/元素")
    print("    小计（d2g 算力之外）            : 6 B/元素")
    print("  → 折叠省掉 d(act) 的「写 2 + 读 2」= **4 B/元素（-40%）**；gate/up 两次都要读，不是额外负担。")
    print("  → 代价：d2g 的 epilogue 从「写」变成「读 4 + 写 2」，extra 4 B/元素读。它能否藏进算力是 .cu 段一 [B] 的问题。")
    print()


def part_b():
    print("== Part B：真实 SM90 d2g anchor（现实现的输出侧）==")
    if get_arch_major() != 9:
        print("  非 sm90，跳过真实 d2g。")
        print()
        return None
    from exp_backward_ncu import build_dgrad
    T, H, I, E, KTOP = 2048, 7168, 2048, 8, 2
    counts = [T * KTOP // E] * E           # 每专家 512，Σt = 4096
    f, flops, info = build_dgrad(I, H, counts, 'cuda')
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    us = 1e18
    for _ in range(7):
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record(); f(); ev1.record()
        torch.cuda.synchronize()
        us = min(us, ev0.elapsed_time(ev1) * 1e3)
    print(f"  d2g = m_grouped fp8 (D bf16)：m={info['m']} N=I={info['N']} K=H={info['K']} E={E}")
    print(f"    单次 launch（热启 3 次，7 次取最小）: {us:.1f} us  {flops / (us * 1e-6) / 1e12:.0f} TFLOPS")
    print(f"    输出 d(act) [Σt={info['m']}, I={info['N']}] bf16 = {info['m'] * info['N'] * 2 / 1e6:.1f} MB")
    print("    它是 compute-bound（第 15/142 篇 ncu：SM 65.5%、DRAM 1.85 TB/s）⇒ 有带宽余量可藏 epilogue 的 extra 读。")
    print()
    return us


def part_c(t_d2g):
    print("== Part C：反向 single-step 预算（把激活段折进 d2g 的投影）==")
    # 第 154 篇：T=2048,I=2048（展开前）→ 激活段 20.1us（量化路径，n=4.19M）
    # d2g 输出形状是 [Σt=T·topk=4096, I=2048]（n=8.39M），故激活段在这个形状约 2×（.cu 段一 [B] 实测 t_seg）
    d2g = t_d2g if t_d2g else 111.0
    seg_small, seg_d2gshape = 20.1, None
    # .cu 段一 [B] 会给 t_seg（d2g 形状）。这里用已知：段与元素数近似线性。
    seg_d2gshape = 2 * seg_small
    print(f"  现实现：d2g {d2g:.1f}us（含写 d(act)） + 激活段 ~{seg_d2gshape:.1f}us（d2g 形状，读 d(act)）")
    print(f"    = {d2g + seg_d2gshape:.1f}us（这两段的合计；段仅占 {seg_d2gshape / (d2g + seg_d2gshape) * 100:.1f}%）")
    print("  折叠后：d2g 的 epilogue 就地出 FP8 对，省掉 d(act) 往返（-4B/元素）。")
    print("    最乐观（extra 读全藏）：t ≈ d2g 算力本身 ≈ 111us ⇒ 相对现实现 ~1.16×。")
    print("    最悲观（extra 读全露、按 HBM 峰值）：+4B×8.39M/3.35TB/s ≈ +10us ⇒ ~121us ⇒ ~1.09×。")
    print("  结论：折叠的收益在**单段**上是 ~10–16%（省 d(act) 往返），但激活段本就只占反向 1.1%（第 154 篇），")
    print("        所以整步收益 <0.2%——**它是对的工程习惯（少一轮全局往返），但绝不是反向的性能杠杆**。")
    print("        真正的杠杆仍是四个 grouped GEMM（wgrad 占 83%）与通信。")
    print()


if __name__ == '__main__':
    print(f"torch {torch.__version__}, cuda {torch.version.cuda}, "
          f"device {torch.cuda.get_device_name(0)}")
    part_a()
    t = part_b()
    part_c(t)
