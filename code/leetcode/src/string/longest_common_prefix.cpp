// 14. 最长公共前缀
// 见 longest_common_prefix.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

std::string longestCommonPrefix(const std::vector<std::string> &strs) {
    if (strs.empty()) return "";
    int firstLen = static_cast<int>(strs[0].size());
    for (int i = 0; i < firstLen; ++i) {
        char ch = strs[0][i];
        for (std::size_t j = 1; j < strs.size(); ++j) {
            if (i == static_cast<int>(strs[j].size()) || strs[j][i] != ch) {
                return strs[0].substr(0, i);
            }
        }
    }
    return strs[0];
}

int main() {
    std::vector<std::string> a = {"flower", "flow", "flight"};
    assert(longestCommonPrefix(a) == "fl");
    std::vector<std::string> b = {"dog", "racecar", "car"};
    assert(longestCommonPrefix(b) == "");
    std::vector<std::string> c = {"abc"};
    assert(longestCommonPrefix(c) == "abc");
    std::vector<std::string> d;
    assert(longestCommonPrefix(d) == "");
    std::vector<std::string> e = {"", "b"};
    assert(longestCommonPrefix(e) == "");
    std::vector<std::string> f = {"ab", "ab", "ab"};
    assert(longestCommonPrefix(f) == "ab");

    std::cout << "longest_common_prefix: all tests passed\n";
    return 0;
}
