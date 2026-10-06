// 451. 根据字符出现频率排序
// 见 sort_characters_by_frequency.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

std::string frequencySort(std::string s) {
    std::unordered_map<char, int> cnt;
    for (char ch : s) {
        ++cnt[ch];
    }
    std::vector<std::pair<char, int>> items(cnt.begin(), cnt.end());
    std::sort(items.begin(), items.end(), [](const auto &a, const auto &b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first < b.first;
    });
    std::string res;
    for (const auto &p : items) {
        res.append(static_cast<size_t>(p.second), p.first);
    }
    return res;
}

int main() {
    assert(frequencySort("tree") == "eert");
    assert(frequencySort("cccaaa") == "aaaccc");
    assert(frequencySort("Aabb") == "bbAa");
    assert(frequencySort("") == "");

    std::cout << "sort_characters_by_frequency: all tests passed\n";
    return 0;
}
