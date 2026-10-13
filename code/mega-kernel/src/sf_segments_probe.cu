// P155-1 (docs/69): MPK SM100 block-scaled FP8 的「SF 段」并置走读的 host 侧算术探针。
//
// device 侧被对走的两个文件（快照 mirage f9eb70c）：
//   * linear_fp8_sm100.cuh        (dense, 单张量线性, 828 行)
//   * fp8_group_gemm_sm100.cuh    (grouped MoE group-GEMM, 1567 行)
//
// 本探针不依赖 SM100 硬件（目标卡 H100/sm90 没有），逐字重放 SF 段的「纯算术」：
//   1) UE8M0 转换     (__float_as_uint(x)>>23)&0xFF
//   2) dense 打包     4 个「不同」per-128-K scale -> uint32，用 (>>8*sf_id)&0xFF 解回
//   3) MoE  打包      1 个 per-128 scale 复制 4 份 -> 4 字节相同 => sf_id 空操作
//   4) 行数账         shape_sfa_k = ceil(K/512) vs K/128
//   5) 转置双射       (i^(lane>>3))*32+lane <-> lane*4+(i^(lane>>3)) 对 4x32
//
// 源码锚点:
//   dense: :78 kNumUTCCPAlignedElems; :83-84 kNumS*StagesPerLoad; :93-94 shape_sf*_k;
//          :159-171 SF TMEM cols; :372-398 TMA SF load; :463-484 UTCCP;
//          :495-500 sfa_id/sfb_id; :536-547 transpose
//   MoE  : :197-198 (4 bytes identical); :197-207 sfa/sfb_smem; :322-336 transpose;
//          :1378/:1399 UE8M0; :1379/:1402 pack; :1232-1233 sfa_id=k_sub
// 生产参数: task_register.cc:4828 (kGranKA=kGranKB=128), :4833 (BLOCK_M=32,BLOCK_N=16,BLOCK_K=128)

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <functional>
#include <set>

static uint32_t f2ue8m0(float x) {
  return (static_cast<uint32_t>(__builtin_bit_cast(uint32_t, x)) >> 23) & 0xFFu;
}

static void hr(const char *t) { printf("\n=== %s ===\n", t); }

int main() {
  hr("1) UE8M0 转换: (__float_as_uint(x)>>23)&0xFF");
  struct { float x; const char *note; } tbl[] = {
      {1.0f, "UE8M0(1.0)=0x7F (MoE SFB padding 默认值)"},
      {2.0f, ""}, {0.5f, ""}, {4.0f, ""}, {0.25f, ""},
      {448.0f, "E4M3 max / 128K block scale 上限附近"}, {1.5f, ""},
  };
  printf("  %-10s %-12s %s\n", "x", "bits>>23&FF", "note");
  for (auto &t : tbl) {
    uint32_t b = __builtin_bit_cast(uint32_t, t.x);
    printf("  %-10.4g 0x%02X (raw 0x%08X)  %s\n", t.x, f2ue8m0(t.x), b, t.note);
  }

  // ------------------------------------------------------------------
  hr("2) dense 打包: 4 个不同 per-128-K scale 装进 1 uint32, sf_id 解回");
  // shape_fa_k = ceil(K/512): 一个 uint32 覆盖 4 个相邻 128-K 块 (gran=128)
  {
    float per128[4] = {0.5f, 2.0f, 0.25f, 4.0f}; // k_block_idx = 0,1,2,3
    uint32_t packed = 0;
    for (int i = 0; i < 4; ++i) packed |= f2ue8m0(per128[i]) << (8 * i);
    printf("  4 个 per-128 scale : %.3g %.3g %.3g %.3g\n", per128[0], per128[1],
           per128[2], per128[3]);
    printf("  host packed uint32 : 0x%08X\n", packed);
    int ok = 1;
    for (uint32_t k_block_idx = 0; k_block_idx < 4; ++k_block_idx) {
      // dense: sfa_id = (kGranKA==32) ? k : k_block_idx % kNumSFAStagesPerLoad
      uint32_t sf_id = k_block_idx % 4; // kNumSFAStagesPerLoad = 4 @ gran 128
      uint32_t byte = (packed >> (8 * sf_id)) & 0xFFu;
      uint32_t want = f2ue8m0(per128[k_block_idx]);
      printf("    k_block=%u -> sf_id=%u byte=0x%02X want=0x%02X %s\n", k_block_idx,
             sf_id, byte, want, byte == want ? "OK" : "MISMATCH");
      ok &= (byte == want);
    }
    printf("  => dense 的 sf_id 是「真地址选择」: %s (4 byte 互不相同)\n",
           ok ? "PASS" : "FAIL");
  }

  // ------------------------------------------------------------------
  hr("3) MoE 打包: 1 个 per-128 scale 复制 4 份, sf_id 空操作");
  {
    float per128 = 2.0f; // fp8_group_gemm: mWeightScale(row, k_tile)
    uint8_t u = f2ue8m0(per128);
    uint32_t packed = u | (u << 8) | (u << 16) | (u << 24); // :1379/:1402
    printf("  per-128 scale = %.3g -> UE8M0 0x%02X\n", per128, u);
    printf("  device packed uint32 = 0x%08X (4 字节全等)\n", packed);
    int same = 1;
    for (int id = 0; id < 4; ++id) same &= (((packed >> (8 * id)) & 0xFFu) == u);
    printf("  => sfa_id=sfb_id=k_sub 选哪一字节都一样: %s\n",
           same ? "PASS (空操作)" : "FAIL");
  }

  // ------------------------------------------------------------------
  hr("4) 行数账: dense shape_sfa_k=ceil(K/512) vs MoE K/128");
  {
    for (uint32_t K : {1024u, 4096u, 7168u}) {
      uint32_t dense = (K + 512 - 1) / 512; // uint32/行 (4xUE8M0)
      uint32_t moe = K / 128;               // fp32/行
      printf("  K=%-6u dense shape_sfa_k=%-4u(uint32/row)  MoE %-4u(fp32/row)  比值=%s\n",
             K, dense, moe, (moe == 4 * dense) ? "4x" : "??");
    }
  }

  // ------------------------------------------------------------------
  hr("5) 转置双射: (i^(lane>>3))*32+lane <-> lane*4+(i^(lane>>3)) 对 4x32");
  {
    // 读: 行 r=i^(lane>>3), 列 lane   -> 线性 r*32+lane
    auto read_lin = [](int i, int lane) { return (i ^ (lane >> 3)) * 32 + lane; };
    // 写: 行 lane, 列 i^(lane>>3)     -> 线性 lane*4 + (i^(lane>>3))
    auto write_lin = [](int i, int lane) { return lane * 4 + (i ^ (lane >> 3)); };
    std::set<int> rd, wr;
    for (int lane = 0; lane < 32; ++lane)
      for (int i = 0; i < 4; ++i) {
        rd.insert(read_lin(i, lane));
        wr.insert(write_lin(i, lane));
      }
    printf("  read 集合 %zu/128, write 集合 %zu/128, 二集合相等=%s\n", rd.size(),
           wr.size(), (rd == wr) ? "YES" : "NO");
    // 「第 g 列写入的是哪些原始行」: write_lin==g 的解 (lane, i)
    for (int g : {0, 1, 17, 63, 64, 127}) {
      printf("  write slot %-3d <- (r,c):", g);
      for (int lane = 0; lane < 32; ++lane)
        for (int i = 0; i < 4; ++i)
          if (write_lin(i, lane) == g)
            printf(" (r=%d,c=%d)", i ^ (lane >> 3), lane);
      printf("\n");
    }
    // XOR 版 (dense/MoE) vs 无 XOR 版 (DeepGEMM 1d1d) 等价性: 读集/写集同
    std::set<int> rd2, wr2; // 无 XOR: read r*32+lane, write lane*4+i
    for (int lane = 0; lane < 32; ++lane)
      for (int i = 0; i < 4; ++i) {
        rd2.insert(i * 32 + lane);
        wr2.insert(lane * 4 + i);
      }
    printf("  无 XOR 版 read 集合=%zu, write 集合=%zu, 与 XOR 版 write 集相等=%s\n",
           rd2.size(), wr2.size(), (wr2 == wr) ? "YES (功能中性置换)" : "NO");
  }

  printf("\n[note] UTCCP/TMEM/UMMA 是 SM100 专属指令, 本机 H100/sm90 未执行;\n"
         "       本文只重放 SMEM 侧的打包/转置算术 (见 docs/59/60/68).\n");
  return 0;
}
