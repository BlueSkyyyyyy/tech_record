// P158-1 复现：UMMA 原语多实现并置审计（host side，纯文本，不 launch kernel）。
//
// 目标：把「同一颗 tcgen05.mma 在几个独立代码库里各封装一遍」这件事做成可复现的表。
// 读取各层的源文件，抽出所有 asm 串里的 `tcgen05.mma ...;` 指令，归一化（去引号/换行/制表、
// 折叠空白），再：
//   1) 打印每层的 distinct opcode（到 operand list 之前）集合；
//   2) 打印全 opcode 矩阵：某 opcode 被哪些层提供；
//   3) 统计每层的能力 flag（.sp 稀疏 / .ws weight-stationary / disable-output-lane mask `{...}` /
//      TMEM-A 操作数 `[%1]` / block_scale）；
//   4) 对四份 `mxf8f6f4.block_scale`（DeepGEMM 自建 / CUTLASS atom / MPK kernel::sm100 /
//      MPK mirage::…::sm100）做逐字相等核对。
//
// 编译：g++ -std=c++17 -O2 src/umma_paths_audit.cu -o /tmp/umma_paths_audit
// 用法：umma_paths_audit <DGHOME> <DGCUTLASS_MMA> <MPK_BLACKWELL_DIR> <MPK_CUTLASS_MMA>
//   例：umma_paths_audit \
//        ~/github/DeepGEMM/deep_gemm/include/deep_gemm/ptx/tcgen05.cuh \
//        ~/github/DeepGEMM/deep_gemm/include/cute/arch/mma_sm100_umma.hpp \
//        /ssd/mirage_mpk_build/include/mirage/persistent_kernel/tasks/blackwell \
//        /ssd/mirage_mpk_build/deps/cutlass/include/cute/arch/mma_sm100_umma.hpp

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static std::string read_file(const std::string &p) {
  std::ifstream f(p);
  if (!f) return std::string();
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// 归一化：把字符串字面量的引号/换行/制表折叠成单空格（相邻字面量自动拼接）。
static std::string normalize(const std::string &t) {
  std::string out;
  out.reserve(t.size());
  bool sp = false;
  for (char c : t) {
    if (c == '"' || c == ' ' || c == '\n' || c == '\r' || c == '\t') {
      if (!sp) { out.push_back(' '); sp = true; }
    } else {
      out.push_back(c);
      sp = false;
    }
  }
  return out;
}

static std::vector<std::string> extract_instrs(const std::string &raw) {
  std::string t = normalize(raw);
  std::regex re(R"(tcgen05\.mma[^;]*;)");
  std::vector<std::string> out;
  for (auto it = std::sregex_iterator(t.begin(), t.end(), re); it != std::sregex_iterator(); ++it) {
    std::string s = it->str();
    // trim
    size_t a = s.find_first_not_of(' '), b = s.find_last_not_of(' ');
    if (a != std::string::npos) out.push_back(s.substr(a, b - a + 1));
  }
  return out;
}

static std::string opcode_of(const std::string &instr) {
  size_t cut = instr.find_first_of("[{");
  std::string s = (cut == std::string::npos) ? instr : instr.substr(0, cut);
  size_t a = s.find_first_not_of(' '), b = s.find_last_not_of(' ');
  return (a == std::string::npos) ? s : s.substr(a, b - a + 1);
}

struct Layer {
  std::string name;
  std::vector<std::string> files;
};

int main(int argc, char **argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <dghome.cuh> <dg_cutlass_mma.hpp> <mpk_blackwell_dir> <mpk_cutlass_mma.hpp>\n", argv[0]);
    return 1;
  }
  std::string dghome = argv[1];
  std::string dgcut  = argv[2];
  std::string mpkdir = argv[3];
  std::string mpkcut = argv[4];

  std::vector<Layer> layers = {
      {"DeepGEMM homegrown ptx::",
       {dghome}},
      {"DeepGEMM vendored CUTLASS atoms",
       {dgcut}},
      {"MPK sm100_ptx (bare fns)",
       {mpkdir + "/sm100_ptx.cuh"}},
      {"MPK kernel::sm100 (struct)",
       {mpkdir + "/sm100_utils.cuh"}},
      {"MPK mirage::...::sm100 (struct)",
       {mpkdir + "/linear_fp8_sm100_sm100_utils.cuh"}},
      {"MPK MLA attention private ptx (tp2/4/8)",
       {mpkdir + "/mla_mtp_decode_tp2_sm100.cuh",
        mpkdir + "/mla_mtp_decode_tp4_sm100.cuh",
        mpkdir + "/mla_mtp_decode_tp8_sm100.cuh"}},
      {"MPK deps/cutlass atoms",
       {mpkcut}},
  };

  std::vector<std::set<std::string>> layer_ops(layers.size());
  std::vector<std::set<std::string>> layer_instrs(layers.size());
  std::vector<std::map<std::string, int>> layer_flags(layers.size());
  std::set<std::string> universe;

  std::printf("== 1. Per-layer distinct tcgen05.mma opcodes ==\n");
  for (size_t li = 0; li < layers.size(); ++li) {
    const Layer &L = layers[li];
    std::vector<std::string> all_instr;
    for (const auto &f : L.files) {
      std::string raw = read_file(f);
      if (raw.empty()) { std::printf("  [warn] cannot read %s\n", f.c_str()); continue; }
      auto v = extract_instrs(raw);
      all_instr.insert(all_instr.end(), v.begin(), v.end());
    }
    for (const auto &i : all_instr) {
      layer_ops[li].insert(opcode_of(i));
      layer_instrs[li].insert(i);
      universe.insert(opcode_of(i));
    }
    std::printf("[%s]  files=%zu  instrs=%zu  distinct_opcodes=%zu\n",
                L.name.c_str(), L.files.size(), all_instr.size(), layer_ops[li].size());
    for (const auto &o : layer_ops[li]) std::printf("    %s\n", o.c_str());
    std::printf("\n");
  }

  std::printf("== 2. Opcode x layer matrix (X = provided) ==\n");
  std::vector<std::string> header_short = {"DGH", "DGC", "M-ptx", "M-k100", "M-mir", "M-mla", "M-cuts"};
  std::printf("  opcode");
  for (auto &h : header_short) std::printf("  %s", h.c_str());
  std::printf("\n");
  int total = 0;
  for (const auto &o : universe) {
    std::printf("  %-52s", o.c_str());
    for (size_t li = 0; li < layers.size(); ++li)
      std::printf("  %-5s", layer_ops[li].count(o) ? "X" : ".");
    std::printf("\n");
    ++total;
  }
  std::printf("  (universe distinct opcodes = %d)\n\n", total);

  std::printf("== 3. Capability flags per layer ==\n");
  for (size_t li = 0; li < layers.size(); ++li) {
    int n_sp = 0, n_ws = 0, n_mask = 0, n_tmemA = 0, n_bscale = 0, n_cg2 = 0, n_cg1 = 0;
    for (const auto &i : layer_instrs[li]) {
      if (i.find(".mma.sp.") != std::string::npos) ++n_sp;
      if (i.find(".mma.ws.") != std::string::npos) ++n_ws;
      if (i.find('{') != std::string::npos) ++n_mask;
      if (i.find("[%1]") != std::string::npos) ++n_tmemA;
      if (i.find("block_scale") != std::string::npos) ++n_bscale;
      if (i.find("cta_group::2") != std::string::npos) ++n_cg2;
      if (i.find("cta_group::1") != std::string::npos) ++n_cg1;
    }
    std::printf("[%s]  sparse.sp=%d  weight-stationary.ws=%d  disable-output-lane{}=%d  TMEM-A[%%1]=%d  block_scale=%d  cta_group::1=%d  cta_group::2=%d\n",
                layers[li].name.c_str(), n_sp, n_ws, n_mask, n_tmemA, n_bscale, n_cg1, n_cg2);
  }
  std::printf("\n");

  // 4. cross-copy equality of the mxf8f6f4.block_scale 1SM / 2SM strings.
  std::printf("== 4. Cross-copy identity of the mxf8f6f4.block_scale wrapper ==\n");
  std::vector<std::string> copies = {
      dghome, dgcut, mpkdir + "/sm100_utils.cuh",
      mpkdir + "/linear_fp8_sm100_sm100_utils.cuh"};
  std::vector<std::string> labels = {"DeepGEMM homegrown", "CUTLASS atom (vendored)", "MPK kernel::sm100", "MPK mirage::...::sm100"};
  for (int cg = 1; cg <= 2; ++cg) {
    std::string needle = "tcgen05.mma.cta_group::";
    needle += std::to_string(cg);
    needle += ".kind::mxf8f6f4.block_scale";
    std::string first;
    int present = 0;
    bool all_same = true;
    std::printf("  cta_group::%d 1SM%s:\n", cg, cg == 2 ? " (2x1SM)" : "");
    for (size_t c = 0; c < copies.size(); ++c) {
      auto v = extract_instrs(read_file(copies[c]));
      std::string hit = "(none)";
      for (const auto &i : v) {
        if (i.rfind(needle, 0) == 0) { hit = i; break; }
      }
      if (hit != "(none)") {
        ++present;
        if (first.empty()) first = hit;
        else if (hit != first) all_same = false;
      }
      std::printf("    %-28s : %s\n", labels[c].c_str(), hit.c_str());
    }
    std::printf("    -> identical among the %d copies that provide it: %s\n",
                present, (present > 0 && all_same) ? "YES" : "NO");
  }
  return 0;
}
