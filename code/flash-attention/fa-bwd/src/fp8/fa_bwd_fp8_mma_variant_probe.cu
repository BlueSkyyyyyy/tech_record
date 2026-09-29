// =============================================================================
// fa_bwd_fp8_mma_variant_probe.cu —— fp8 `mma.m16n8k32` 合法布局探针（F6 判定的 ISA 依据）
// =============================================================================
// 目的：确认 fp8 `mma.sync.aligned.m16n8k32` 在 sm90a 上**只支持 `.row.col`**。
//   - 合法：`mma...m16n8k32.row.col.f32.e4m3.e4m3.f32`（本文件编译这条）。
//   - 非法：`.col.row` / `.row.row` / `.col.col` —— ptxas 报
//     `Illegal alayout '.col' for instruction 'mma'` / `Illegal blayout '.row' ...`，
//     原始 ptxas 输出见同目录 `fa_bwd_fp8_mma_variant_probe.illegal.out.txt`。
// 结论：B 操作数**必须是 col-major**（逻辑 B[K][N] 存成 [N][K]、K 连续）⇒ 反向的
//   GEMM3/4/5（B=Qᵀ/dOᵀ/Kᵀ）**必须转置**，不能沿用 GEMM1/2 的 K-major（N 连续）tile。
//
// 运行：scripts/run.sh src/fp8/fa_bwd_fp8_mma_variant_probe.cu
// =============================================================================

#include <cstdint>
#include <cstdio>

__global__ void probe(const uint32_t* a, const uint32_t* b, float* c) {
  uint32_t ar[4] = {a[0], a[1], a[2], a[3]};
  uint32_t br[2] = {b[0], b[1]};
  float d[4] = {0, 0, 0, 0};
  // 唯一被 sm90a 接受的 fp8 m16n8k32 布局：row.col。
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]), "r"(br[0]), "r"(br[1]));
  c[threadIdx.x] = d[0] + d[1] + d[2] + d[3];
}

int main() {
  printf("=== fp8 mma.m16n8k32 布局探针 ===\n");
  printf("  row.col : 编译通过（fp8 m16n8k32 唯一合法布局；B 必须 col-major）\n");
  printf("  col.row / row.row / col.col : ptxas 拒绝（见 .illegal.out.txt）\n");
  printf("=== PASS（.row.col 可用；转置不可免） ===\n");
  return 0;
}
