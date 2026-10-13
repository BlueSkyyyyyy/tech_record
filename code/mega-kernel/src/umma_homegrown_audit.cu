// P160-1 复现：DeepGEMM 自建 PTX UMMA 封装层逐条对照 + 两个 helper + MPK 手写 tcgen05_mma/commit。
//
// 目标（host side，纯文本，不 launch kernel——SM100 本体本机不可执行）：
//   把 docs/72（只对齐 tcgen05.mma 指令串集合）与 docs/73（只对齐 SMEM 描述符构造）之间
//   剩下的三面补齐：
//     A) DeepGEMM homegrown `ptx/tcgen05.cuh` 的 9 个 struct「除 mxf8f6f4.block_scale 外」
//        与 CUTLASS `mma_sm100_umma.hpp` 对应 atom 逐条对照——尤其是 dense 形态去掉的
//        disable-output-lane 掩码 `{%5,%6,%7,%8}`（docs/72 §5.2 的裁剪口径）与 elect 守卫；
//     B) 两个 helper：`tmem_load_32dp32b<K>` 的 conditional_t 分派 / `umma_arrive_no_elect`
//        （无内部 elect）对 CUTLASS 的 `SM100_TMEM_LOAD_32dp32b{4..128}x` / `umma_arrive`（内部 elect）；
//     C) MPK 手写的 `sm100_ptx::tcgen05_mma`（kind::f16）/`tcgen05_commit` 与
//        DeepGEMM homegrown / CUTLASS 对应物的操作数形状并置。
//
// 编译：g++ -x c++ -std=c++17 -O2 src/umma_homegrown_audit.cu -o /tmp/umma_homegrown_audit
// 用法：umma_homegrown_audit <dghome.cuh> <dg_cutlass_mma.hpp> <mpk_sm100_ptx.cuh>
//                            <cutlass_copy_sm100.hpp> <cutlass_barrier.h>
//                            <mpk_sm100_utils.cuh> <mpk_fp8_sm100_utils.cuh>

#include <algorithm>
#include <cctype>
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

// 从归一化文本抽所有 `tcgen05.mma...;` 指令（不含 commit）。
static std::vector<std::string> extract_mma(const std::string &raw) {
  std::string t = normalize(raw);
  std::regex re(R"(tcgen05\.mma[^;]*;)");
  std::vector<std::string> out;
  for (auto it = std::sregex_iterator(t.begin(), t.end(), re); it != std::sregex_iterator(); ++it) {
    std::string s = it->str();
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

// 去掉 CUTLASS dense 形态恒传的全零 disable-output-lane 掩码 `, {%5, %6, %7, %8}`。
static std::string strip_mask(const std::string &instr) {
  std::regex re(R"(, \{%[^}]*\})");
  return std::regex_replace(instr, re, "");
}

static int operand_count(const std::string &instr) {
  // 数 `%` 出现次数（每个占位操作数一次）。
  int n = 0;
  for (size_t i = 0; i + 1 < instr.size(); ++i)
    if (instr[i] == '%' && instr[i + 1] >= '0' && instr[i + 1] <= '9') ++n;
  return n;
}

// 从原文找出 `struct NAME`（NAME 后须是非标识符字符，避免匹配 *_SPARSE 等更长名）到下一个 `\n};` 的 body。
static std::string struct_body(const std::string &text, const std::string &name) {
  std::string needle = "struct " + name;
  size_t p = 0;
  while ((p = text.find(needle, p)) != std::string::npos) {
    size_t after = p + needle.size();
    char c = (after < text.size()) ? text[after] : '\0';
    if (!(std::isalnum((unsigned char)c) || c == '_')) break;
    p = after;
  }
  if (p == std::string::npos) return std::string();
  size_t e = text.find("\n};", p);
  if (e == std::string::npos) e = text.size();
  return text.substr(p, e - p);
}

// 从归一化文本里，找到含 needle 的那条指令（到下一个 ';'）。
static std::string instr_containing(const std::string &raw, const std::string &needle) {
  std::string t = normalize(raw);
  size_t p = t.find(needle);
  if (p == std::string::npos) return std::string("(none)");
  size_t e = t.find(';', p);
  if (e == std::string::npos) e = t.size();
  std::string s = t.substr(p, e - p + 1);
  size_t a = s.find_first_not_of(' '), b = s.find_last_not_of(' ');
  return (a == std::string::npos) ? s : s.substr(a, b - a + 1);
}

static bool contains(const std::string &hay, const std::string &needle) {
  return hay.find(needle) != std::string::npos;
}

// 去所有空白（便于跨字面量拆分差异比较 asm）。
static std::string squash(const std::string &s) {
  std::string out;
  for (char c : s) if (c != ' ') out.push_back(c);
  return out;
}

struct CAtom {
  std::string name;
  std::string body;
  std::vector<std::string> instrs;
  bool elect = false;
  bool mask = false;
};

int main(int argc, char **argv) {
  if (argc < 8) {
    std::fprintf(stderr, "usage: %s <dghome> <dg_cut_mma> <mpk_ptx> <copy_sm100> <barrier.h> <mpk_utils> <mpk_fp8_utils>\n", argv[0]);
    return 1;
  }
  std::string dghome = argv[1], dgcut = argv[2], mpkptx = argv[3];
  std::string copy = argv[4], barrier = argv[5];
  std::string mpkutils = argv[6], mpkfp8 = argv[7];

  std::string dgh = read_file(dghome);
  std::string dgc = read_file(dgcut);
  std::string mpt = read_file(mpkptx);
  std::string cpy = read_file(copy);
  std::string bar = read_file(barrier);
  std::string mut = read_file(mpkutils);
  std::string mfp = read_file(mpkfp8);
  for (auto &pr : {std::make_pair("dghome", &dgh), std::make_pair("dgcut", &dgc),
                   std::make_pair("mpkptx", &mpt), std::make_pair("copy", &cpy),
                   std::make_pair("barrier", &bar), std::make_pair("mpkutils", &mut),
                   std::make_pair("mpkfp8", &mfp)}) {
    if (pr.second->empty()) std::printf("[warn] cannot read %s\n", pr.first);
  }

  // 枚举 CUTLASS 所有 SM100_MMA_* struct。
  std::vector<CAtom> atoms;
  {
    std::set<std::string> seen;
    std::regex re(R"(struct (SM100_MMA_[A-Za-z0-9_]+))");
    for (auto it = std::sregex_iterator(dgc.begin(), dgc.end(), re); it != std::sregex_iterator(); ++it) {
      std::string name = (*it)[1].str();
      if (!seen.insert(name).second) continue;
      CAtom a;
      a.name = name;
      a.body = struct_body(dgc, name);
      a.instrs = extract_mma(a.body);
      a.elect = contains(a.body, "elect_one_sync");
      for (auto &i : a.instrs) if (contains(i, "{")) a.mask = true;
      atoms.push_back(a);
    }
  }

  const std::vector<std::string> home_names = {
      "SM100_MMA_F16BF16_SS",        "SM100_MMA_F16BF16_2x1SM_SS",
      "SM100_MMA_MXF8F6F4_SS",       "SM100_MMA_MXF8F6F4_2x1SM_SS",
      "SM100_MMA_F8F6F4_SS",         "SM100_MMA_F8F6F4_2x1SM_SS",
      "SM100_MMA_MXF4_SS",           "SM100_MMA_MXF4_2x1SM_SS",
      "SM100_MMA_F16BF16_WS_SS"};

  std::printf("== A. DeepGEMM homegrown ptx:: struct  ->  CUTLASS atom (逐条对照) ==\n");
  std::printf("  %-30s %-46s %-34s %-6s %-6s %-8s %-11s\n",
              "homegrown struct", "opcode", "CUTLASS counterpart", "elect", "mask", "verbatim", "after-strip");
  for (const auto &hn : home_names) {
    std::string hb = struct_body(dgh, hn);
    auto hi = extract_mma(hb);
    if (hi.empty()) { std::printf("  %-30s (no tcgen05.mma instr found)\n", hn.c_str()); continue; }
    std::string h = hi[0];
    std::string hop = opcode_of(h);
    // 找 CUTLASS 对端：strip_mask 后逐字相等者优先。
    std::string match_struct = "(none)", verbatim = "N", after = "N";
    std::string c_elect = "-", c_mask = "-";
    for (const auto &a : atoms) {
      for (const auto &ci : a.instrs) {
        if (ci == h) {
          if (match_struct == "(none)") { match_struct = a.name; verbatim = "Y"; after = "Y";
            c_elect = a.elect ? "Y" : "N"; c_mask = a.mask ? "Y" : "N"; }
        } else if (strip_mask(ci) == h && match_struct == "(none)") {
          match_struct = a.name; after = "Y";
          c_elect = a.elect ? "Y" : "N"; c_mask = a.mask ? "Y" : "N";
        }
      }
    }
    std::printf("  %-30s %-46s %-34s %-6s %-6s %-8s %-11s\n",
                hn.c_str(), hop.c_str(), match_struct.c_str(),
                c_elect.c_str(), c_mask.c_str(), verbatim.c_str(), after.c_str());
  }
  std::printf("\n  ## dense f16 逐字样例（homegrown 去掉掩码后 == CUTLASS）：\n");
  std::printf("     homegrown : %s\n", instr_containing(dgh, "tcgen05.mma.cta_group::1.kind::f16").c_str());
  std::printf("     CUTLASS   : %s\n", instr_containing(dgc, "tcgen05.mma.cta_group::1.kind::f16").c_str());
  std::printf("     stripped? : mask {%%...} removed -> homegrown [%s] CUTLASS [%s]\n",
              strip_mask(instr_containing(dgh, "tcgen05.mma.cta_group::1.kind::f16")).c_str(),
              strip_mask(instr_containing(dgc, "tcgen05.mma.cta_group::1.kind::f16")).c_str());
  std::printf("\n  ## operand 计数（%% 占位符个数）homegrown vs 首个 CUTLASS 同 opcode：\n");
  for (const auto &hn : home_names) {
    std::string hb = struct_body(dgh, hn);
    auto hi = extract_mma(hb);
    if (hi.empty()) continue;
    std::string hop = opcode_of(hi[0]);
    int hn_oper = operand_count(hi[0]);
    int cn_oper = -1;
    for (const auto &a : atoms)
      for (const auto &ci : a.instrs)
        if (opcode_of(ci) == hop) { cn_oper = operand_count(ci); break; }
    std::printf("     %-30s opcode=%-44s operands homegrown=%d cutlass=%d\n",
                hn.c_str(), hop.c_str(), hn_oper, cn_oper);
  }

  std::printf("\n== B. helper 对照 ==\n");
  // tmem_load_32dp32b dispatch
  {
    std::printf("  [tmem_load_32dp32b] DeepGEMM 分派表 (kNumValues -> CUTLASS struct)：\n");
    std::regex re(R"(kNumValues == (\d+), cute::(SM100_TMEM_LOAD_32dp32b\d+x))");
    std::string t = normalize(dgh);
    for (auto it = std::sregex_iterator(t.begin(), t.end(), re); it != std::sregex_iterator(); ++it) {
      std::string k = (*it)[1].str();
      std::string s = (*it)[2].str();
      bool present = contains(normalize(cpy), "struct " + s);
      std::printf("     kNumValues=%-4s -> %-32s exists in copy_sm100.hpp: %s\n",
                  k.c_str(), s.c_str(), present ? "Y" : "N");
    }
    // 最后一个 128x 是 conditional_t 链的 else 分支（无 `kNumValues ==` 前缀）。
    std::printf("     kNumValues=%-4s -> %-32s exists in copy_sm100.hpp: %s (else branch, no `==` prefix)\n",
                "128", "SM100_TMEM_LOAD_32dp32b128x",
                contains(normalize(cpy), "struct SM100_TMEM_LOAD_32dp32b128x") ? "Y" : "N");
    // 也打印拷贝里能匹配的 struct 数
    std::regex re2(R"(struct (SM100_TMEM_LOAD_32dp32b(\d+)x))");
    std::set<std::string> got;
    for (auto it = std::sregex_iterator(cpy.begin(), cpy.end(), re2); it != std::sregex_iterator(); ++it)
      got.insert((*it)[1].str());
    std::printf("     copy_sm100.hpp 里 SM100_TMEM_LOAD_32dp32b<M>x（含 _16b）struct 数：%zu\n", got.size());
  }
  // umma_arrive_no_elect vs CUTLASS umma_arrive
  {
    std::printf("  [umma_arrive] 内部 elect 守卫对照：\n");
    // 用所在函数体近似：取 umma_arrive_no_elect 之后 600 字符
    size_t p = dgh.find("umma_arrive_no_elect");
    std::string win = (p == std::string::npos) ? std::string() : dgh.substr(p, 600);
    std::printf("     DeepGEMM umma_arrive_no_elect : body has elect_one_sync = %s\n",
                contains(win, "elect_one_sync") ? "Y" : "N");
    p = bar.find("void umma_arrive(uint64_t");
    std::string win2 = (p == std::string::npos) ? std::string() : bar.substr(p, 600);
    std::printf("     CUTLASS umma_arrive          : body has elect_one_sync = %s\n",
                contains(win2, "elect_one_sync") ? "Y" : "N");
    p = bar.find("void umma_arrive_multicast_no_elect");
    std::string win3 = (p == std::string::npos) ? std::string() : bar.substr(p, 600);
    std::printf("     CUTLASS umma_arrive_multicast_no_elect : body has elect_one_sync = %s\n",
                contains(win3, "elect_one_sync") ? "Y" : "N");
    std::printf("     DeepGEMM arrive asm : %s\n",
                instr_containing(dgh, "tcgen05.commit.cta_group::1.mbarrier::arrive::one").c_str());
    std::printf("     CUTLASS  arrive asm : %s\n",
                instr_containing(bar, "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64").c_str());
  }
  // tcgen05.fence / wait
  {
    auto count = [](const std::string &s, const std::string &needle) {
      int n = 0; size_t p = 0;
      while ((p = s.find(needle, p)) != std::string::npos) { ++n; p += needle.size(); }
      return n;
    };
    std::printf("  [fence] `tcgen05.fence::before/after_thread_sync` 出现次数：\n");
    std::printf("     DeepGEMM ptx/tcgen05.cuh : %d / %d\n",
                count(dgh, "tcgen05.fence::before_thread_sync"), count(dgh, "tcgen05.fence::after_thread_sync"));
    std::printf("     MPK sm100_ptx.cuh        : %d / %d\n",
                count(mpt, "before_thread_sync"), count(mpt, "after_thread_sync"));
    std::printf("     MPK sm100_utils.cuh      : %d / %d\n",
                count(mut, "before_thread_sync"), count(mut, "after_thread_sync"));
    std::printf("     MPK fp8_sm100_utils.cuh  : %d / %d\n",
                count(mfp, "before_thread_sync"), count(mfp, "after_thread_sync"));
    std::printf("     CUTLASS barrier.h        : %d / %d (含任何位置)\n",
                count(bar, "before_thread_sync"), count(bar, "after_thread_sync"));
    std::printf("     CUTLASS mma_sm100_umma   : %d / %d\n",
                count(dgc, "before_thread_sync"), count(dgc, "after_thread_sync"));
  }

  std::printf("\n== C. MPK 手写 sm100_ptx 与 DeepGEMM/CUTLASS 并置 ==\n");
  {
    std::string mpk_mma = instr_containing(mpt, "tcgen05.mma.cta_group::1.kind::f16");
    std::string dg_f16 = instr_containing(dgh, "tcgen05.mma.cta_group::1.kind::f16");
    std::string cu_f16 = instr_containing(dgc, "tcgen05.mma.cta_group::1.kind::f16");
    std::printf("  MPK sm100_ptx::tcgen05_mma  : %s\n", mpk_mma.c_str());
    std::printf("  DeepGEMM F16BF16_SS         : %s\n", dg_f16.c_str());
    std::printf("  CUTLASS  F16BF16_SS         : %s\n", cu_f16.c_str());
    std::printf("  -> MPK == DeepGEMM (raw): %s ;  MPK == strip_mask(CUTLASS): %s\n",
                (mpk_mma == dg_f16) ? "Y" : "N", (mpk_mma == strip_mask(cu_f16)) ? "Y" : "N");
    std::string mpk_cmt = instr_containing(mpt, "tcgen05.commit.cta_group::1.mbarrier::arrive::one");
    std::string dg_cmt = instr_containing(dgh, "tcgen05.commit.cta_group::1.mbarrier::arrive::one");
    std::string cu_cmt = instr_containing(bar, "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64");
    std::printf("  MPK sm100_ptx::tcgen05_commit : %s\n", mpk_cmt.c_str());
    std::printf("  DeepGEMM umma_arrive_no_elect : %s\n", dg_cmt.c_str());
    std::printf("  CUTLASS  umma_arrive          : %s\n", cu_cmt.c_str());
    std::printf("  -> MPK commit == DeepGEMM (squashed): %s ;  MPK commit == CUTLASS asm (squashed): %s\n",
                (squash(mpk_cmt) == squash(dg_cmt)) ? "Y" : "N",
                (squash(mpk_cmt) == squash(cu_cmt)) ? "Y" : "N");
  }

  return 0;
}
