// =============================================================================
// fa_bwd_dump.h —— P3-3：把 ours 反向输出落成 CPU npy，供 harness/fa_bwd_compare.py
//   统一做「ours vs ref vs FA/TE」对拍汇总。
// =============================================================================
// 只依赖标准库；被 6 个 host（fp16/bf16/fp8 × 单文件/两文件）以 `#include "../fa_bwd_dump.h"`
// 引入。写 1D little-endian C-contiguous float32 npy（与 harness 读 ref_*.npy 的 `<f4` 一致）。
//   * 头部布局：magic(6) + version(1,0)(2) + HEADER_LEN(2, little-endian) + header；
//   * header 空格补齐、结尾 `\n`，使 `10 + len(header)` 为 64 的整数倍（对齐 numpy 写出口径）。
#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

inline void fa_bwd_save_npy_f32(const std::string& path, const float* data, long long n) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "fa_bwd_save_npy_f32: 无法写入 %s\n", path.c_str());
    return;
  }
  const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
  f.write(magic, 6);
  const unsigned char ver[2] = {1, 0};
  f.write(reinterpret_cast<const char*>(ver), 2);

  char raw[128];
  int hlen = std::snprintf(raw, sizeof(raw),
                           "{'descr': '<f4', 'fortran_order': False, 'shape': (%lld,), }", n);
  std::string h(raw, (size_t)(hlen > 0 ? hlen : 0));
  // 使 10 + h.size() 为 64 的整数倍，且至少补 1 个字符（末尾换行）。
  int pad = (int)((64 - (10 + (int)h.size()) % 64) % 64);
  if (pad == 0) pad = 64;
  h.append((size_t)(pad - 1), ' ');
  h.push_back('\n');

  const uint16_t hl = (uint16_t)h.size();
  f.write(reinterpret_cast<const char*>(&hl), 2);
  f.write(h.data(), (std::streamsize)h.size());
  f.write(reinterpret_cast<const char*>(data), (std::streamsize)(n * 4));
}
