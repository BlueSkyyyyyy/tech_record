#!/usr/bin/env python3
# P125-1 实验：真实形状（H=7168 / hc=4）下 mHC 反向单 kernel 接线的账目。
#   - 直接从真实头文件 layout/mega_mhc.cuh 解析常量（防漂移）
#   - 复刻前向 Post worker 的 lane 网格与每 lane 工作量
#   - TMEM 列预算（与 device 侧同式）
#   - 「G0 累加器留在 TMEM 直喂 K1」相对「先落 global scratch 再读回」省掉的往返字节
#   - K1 阶段的算术强度（元素级）与 H100 roofline 归属
import re, os, math, json

INC = "~/github/DeepGEMM/deep_gemm/include"
LAYOUT = os.path.join(INC, "deep_gemm/layout/mega_mhc.cuh")
src = open(LAYOUT).read()

def const(name):
    m = re.search(r"\b" + name + r"\s*=\s*([^;]+);", src)
    expr = m.group(1).strip()
    # 只支持算术表达式（kNumHCOutputs = kNumRoutes*(kNumRoutes+2) 等）
    kNumRoutes_ = int(re.search(r"kNumRoutes\s*=\s*(\d+)", src).group(1))
    env = {"kNumRoutes": kNumRoutes_, "kNumHCOutputs": kNumRoutes_ * (kNumRoutes_ + 2)}
    return int(eval(expr, {}, env))

kNumRoutes = int(re.search(r"kNumRoutes\s*=\s*(\d+)", src).group(1))
kNumHCOutputs = eval(re.search(r"kNumHCOutputs\s*=\s*(.+?);", src).group(1), {}, {"kNumRoutes": kNumRoutes})
BLOCK_M = const("BLOCK_M"); BLOCK_N = const("BLOCK_N"); BLOCK_K = const("BLOCK_K")
kNumWarpsPerWG = const("kNumWarpsPerWG")
assert (kNumRoutes, kNumHCOutputs, BLOCK_M, BLOCK_N, BLOCK_K, kNumWarpsPerWG) == (4, 24, 64, 24, 64, 4), "header drift"

UMMA_K = 32 // 4           # 32/sizeof(float)
kNumPostWGs = 2
kNumTokenRowsPerLane = 2
kNumRowsPerWarp = BLOCK_M // kNumWarpsPerWG
kNumRowGroupsPerWarp = kNumRowsPerWarp // kNumTokenRowsPerLane
kNumLanesPerRowGroup = 32 // kNumRowGroupsPerWarp

def align_tmem(c):
    for g in (32, 64, 128, 256, 512):
        if c <= g: return g
    raise ValueError

print("== 头文件解析（%s）==" % LAYOUT)
print("  kNumRoutes=%d kNumHCOutputs=%d BLOCK_M=%d BLOCK_N=%d BLOCK_K=%d kNumWarpsPerWG=%d"
      % (kNumRoutes, kNumHCOutputs, BLOCK_M, BLOCK_N, BLOCK_K, kNumWarpsPerWG))
print("  UMMA_K=%d kNumPostWGs=%d kNumRowsPerWarp=%d kNumRowGroupsPerWarp=%d kNumLanesPerRowGroup=%d"
      % (UMMA_K, kNumPostWGs, kNumRowsPerWarp, kNumRowGroupsPerWarp, kNumLanesPerRowGroup))

# 真实形状
H, hc, T = 7168, 4, 4096
NOUT = hc * (hc + 2)
D = hc * H
splits = 16
num_m_blocks = math.ceil(T / BLOCK_M)
num_tasks = num_m_blocks * splits
atoms = H // UMMA_K
k_blocks = H // BLOCK_K
atoms_per_kblock = BLOCK_K // UMMA_K

print("\n== 真实形状（DeepSeek 量级 H=%d hc=%d D=%d T=%d splits=%d）==" % (H, hc, D, T, splits))
print("  atoms=H/UMMA_K=%d  k_blocks=H/BLOCK_K=%d  atoms/k-block=%d" % (atoms, k_blocks, atoms_per_kblock))
print("  num_m_blocks=%d  num_tasks=num_m_blocks*splits=%d" % (num_m_blocks, num_tasks))
print("  每个 Post worker 的原子数 = %d（两 worker 各半）；每 lane 覆盖 %d 浮点对/行/原子"
      % (atoms // kNumPostWGs, UMMA_K // 2))
floats_per_worker_per_row = (atoms // kNumPostWGs) * UMMA_K      # 单 route
print("  每 lane 每行 每 route 累加 %.0f floats；×%d routes = %.0f floats/行（一个 worker）"
      % (floats_per_worker_per_row, kNumRoutes, floats_per_worker_per_row * kNumRoutes))
print("  逐行归约路径：4-lane(row-group) 内 `warp_reduce_sum<4>` → 写 smem[worker][row] → 两 worker 相加")
print("  （每行由 %d worker × %d lane = %d lane 负责，与 device A 对拍一致）"
      % (kNumPostWGs, kNumLanesPerRowGroup, kNumPostWGs * kNumLanesPerRowGroup))

# TMEM 预算
A_stage = kNumRoutes * UMMA_K
accum_start = 4 * A_stage          # kNumAStages=4
fwd_used = accum_start + 2 * BLOCK_N   # kNumAccumStages=2
fwd_cols = align_tmem(fwd_used)
g0_cols = align_tmem(BLOCK_N)
print("\n== TMEM 列预算（整块 512 列）==")
print("  每 A stage 列=%d accum 起点=%d 前向 used=%d -> %d" % (A_stage, accum_start, fwd_used, fwd_cols))
print("  反向 G0 累加器 u0[%d,%d] used=%d -> %d" % (BLOCK_M, BLOCK_N, BLOCK_N, g0_cols))
print("  两者之和 %d <= 512: %s" % (fwd_cols + g0_cols, "OK" if fwd_cols + g0_cols <= 512 else "FAIL"))
print("  dz[T,D] = %d floats/行 远超 512 列 -> 只能按 K-chunk 流式（同前向 A stage 的 per-k-block 粒度）" % D)

# 「G0 直喂 K1」省掉的 u0 往返
u0_elems = num_m_blocks * splits * BLOCK_M * NOUT
u0_bytes_rt = u0_elems * 4 * 2     # 写 + 读
dfn_bytes = NOUT * D * 4
print("\n== 「G0 累加器直喂 K1」 vs 「落 global scratch 再读回」==")
print("  前向 WG3 的 gemm_partial = num_m_blocks*splits*BLOCK_M*NOUT = %d elems = %.3f MB"
      % (u0_elems, u0_elems * 4 / 1e6))
print("  省掉的往返（写+读）= %.3f MB  （相对 dfn=%d elems=%.3f MB 很小）"
      % (u0_bytes_rt / 1e6, NOUT * D, dfn_bytes / 1e6))
print("  => 「直喂」的主要收益不是省字节，而是**省掉 num_tasks 次 global round-trip 的延迟**"
      "（每 task 一个 split 的 24 列累加器，跨 WG 同步后立刻被同 CTA 的 K1 消费）。")

# K1 算术强度（元素级）
# K1 读：A[T,D] + dS[T,D] + fn[NOUT,D]（G1a 复用）+ dY[T,H] + 少量小张量；写：du0[T,NOUT] + 部分小张量。
read_bytes = (T * D * 2 + T * H + NOUT * D) * 4
write_bytes = (T * NOUT + T * H + T * hc * hc + T * hc + T * hc) * 4
flops = (T * D * 4    # dz = Σ du0[k] fn[k]（G1a 折进 K1）4 次/元素
         + T * D * 2  # dA = dAbase + dz 等元素级
         + T * H * 6) # RMSNorm 反向
ai = flops / (read_bytes + write_bytes)
print("\n== K1（含 G1a 折入）元素级算术强度 ==")
print("  read=%.1f MB write=%.1f MB flops~%.1f MFLOP -> AI=%.2f FLOP/B" %
      (read_bytes / 1e6, write_bytes / 1e6, flops / 1e6, ai))
print("  H100 HBM 实测可达 ~3.3 TB/s，FP32 峰值 ~67 TFLOPS -> 拐点 ~20 FLOP/B；"
      "AI=%.2f -> 访存受限" % ai)

# device D 段实测的带宽（来自 CUDA 输出，H=1024 时约 657-888 GB/s），真实 D=28672 会更高。
print("\n== 结论 ==")
print("  1) 前向 Post 网格（2 worker × 每行 4 lane、4-lane 归约）在 device 上与 host double 一致（rel 5e-8）；")
print("     用 32-lane 全 warp 归约会引入 rel ~2.47 的**结构性错误**（不是数值误差）。")
print("  2) TMEM 预算：前向 256 列 + G0 累加器 32 列 = 288 < 512，单 kernel 内共存可行。")
print("  3) K1+G1a 是访存受限的元素级段，与第 131/132 篇的六段账一致；目标卡上应复用前向 Post/Normal worker。")
