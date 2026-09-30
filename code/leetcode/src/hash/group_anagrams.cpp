// 49. 字母异位词分组（哈希表 + 排序指纹）
// 见 group_anagrams.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

std::vector<std::vector<std::string>> groupAnagrams(std::vector<std::string> strs) {
    std::unordered_map<std::string, std::vector<std::string>> groups;
    for (const std::string &s : strs) {
        std::string key = s;
        std::sort(key.begin(), key.end());
        groups[key].push_back(s);
    }
    std::vector<std::vector<std::string>> res;
    for (auto &pair : groups) res.push_back(pair.second);
    return res;
}

int main() {
    std::vector<std::vector<std::string>> got =
        groupAnagrams({"eat", "tea", "tan", "ate", "nat", "bat"});
    for (auto &g : got) std::sort(g.begin(), g.end());
    std::sort(got.begin(), got.end());
    std::vector<std::vector<std::string>> want = {
        {"ate", "eat", "tea"}, {"bat"}, {"nat", "tan"}};
    assert(got == want);

    std::vector<std::vector<std::string>> one = groupAnagrams({""});
    assert(one.size() == 1 && one[0] == std::vector<std::string>{""});

    std::cout << "group_anagrams: all tests passed\n";
    return 0;
}
