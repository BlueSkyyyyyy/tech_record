// MN-major（转置）SW128 GMMA 描述符 + 布局辅助。
//
// 用途：MLA 的 PV 需要 B=V，形状 (N=DV, K=KT)，其中 global 的 c_kv 是 [KT][DC]（DC 连续）。
// 若把 V 按「MN-major」喂给 wgmma（N 维连续），就不用再做昂贵的转置/非合并 gather。
//
// 依据 CUTLASS `mma_traits_sm90_gmma.hpp` 的 canonical MN-major B128 布局（单位 uint128_t）：
//   LayoutType::B128 : Swizzle<3,4,3> o smem_ptr o ((8,n),(8,k)):((1,LBO),(8,SBO))
// 即物理布局 = [k/8][n/64][8 行][64 元素]，每 atom 1024B；行内 16B 列做 c' = c ^ r（r = 行号 %8）。
//   - 「行」沿 K（kt），「列」沿 N（dv）。
//   - LBO = 相邻 n-group（每 64 个 N 元素）的字节距离 = 1024（8 个 n-group 不跨，恒 1024）。
//   - SBO = 相邻 k-group（每 8 个 K 行）的字节距离 = (N/64)*1024（因为 n-group 在 k-group 内层）。
//   - k16 块 s 的起始 = 第 2s 个 k-group → 偏移 2s*SBO。
//
// 与 K-major 的区别：K-major 的 SBO 沿「8 行组」是 (K/64)*1024；MN-major 的 LBO/SBO 语义对调。
#pragma once

// 元素 (kt, n) -> MN-major SW128 字节偏移；N = 该 operand tile 的 N 宽度（64 的倍数）。
__device__ __forceinline__ int mn128_off(int kt, int n, int N) {
  const int kg = kt >> 3, kr = kt & 7;
  const int ng = n >> 6, c = (n >> 3) & 7, cc = c ^ kr;
  return kg * (N >> 6) * 1024 + ng * 1024 + kr * 128 + cc * 16 + (n & 7) * 2;
}

// 存一个 16B（8 个 bf16）到 MN-major SW128 tile：要求 n0 % 8 == 0。
__device__ __forceinline__ void mn128_store16(char* tile, int kt, int n0, int N, uint4 v) {
  const int kg = kt >> 3, kr = kt & 7;
  const int ng = n0 >> 6, c = (n0 >> 3) & 7, cc = c ^ kr;
  *reinterpret_cast<uint4*>(tile + kg * (N >> 6) * 1024 + ng * 1024 + kr * 128 + cc * 16) = v;
}

// MN-major SW128 描述符：addr = 本 k16 块 (kt=16s, n=0) 的字节地址；sbo = 相邻 8 个 K 行的字节距离。
__device__ __forceinline__ uint64_t make_desc_mn128(uint32_t addr, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((1024u >> 4) & 0x3FFF) << 16;  // LBO = 64（16B 单位），n-group 步长 1024B
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)0 << 49;
  d |= (uint64_t)1 << 62;  // B128
  return d;
}

// k16 块索引 s（沿 K）对应的描述符地址增量：块 s 从第 2s 个 k-group 开始。
__device__ __forceinline__ uint32_t mn128_k16_addr(uint32_t base, int s, int N) {
  return base + (uint32_t)(s * 2 * ((N >> 6) * 1024));
}
