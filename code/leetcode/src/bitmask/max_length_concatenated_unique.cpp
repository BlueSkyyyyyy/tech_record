// 1239. 串联字符串的最大长度
// 见 max_length_concatenated_unique.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

int maxLength(const std::vector<std::string> &arr) {
    std::vector<std::pair<int, int>> items;
    for (const std::string &s : arr) {
        int mask = 0;
        bool ok = true;
        for (char ch : s) {
            int bit = 1 << (ch - 'a');
            if (mask & bit) {
                ok = false;
                break;
            }
            mask |= bit;
        }
        if (ok) {
            items.push_back({mask, static_cast<int>(s.size())});
        }
    }

    std::unordered_map<int, int> dp;
    dp[0] = 0;
    for (auto &[mask, length] : items) {
        std::vector<std::pair<int, int>> snapshot(dp.begin(), dp.end());
        for (auto &[cur, total] : snapshot) {
            if ((cur & mask) == 0) {
                int merged = cur | mask;
                int cand = total + length;
                auto it = dp.find(merged);
                if (it == dp.end() || it->second < cand) {
                    dp[merged] = cand;
                }
            }
        }
    }
    int best = 0;
    for (auto &[k, v] : dp) {
        best = std::max(best, v);
    }
    return best;
}

int main() {
    assert(maxLength({"un", "iq", "ue"}) == 4);
    assert(maxLength({"cha", "r", "act", "ers"}) == 6);
    assert(maxLength({"abcdefghijklmnopqrstuvwxyz"}) == 26);
    assert(maxLength({"aa", "bb"}) == 0);
    assert(maxLength({"a", "b", "c"}) == 3);

    std::cout << "max_length_concatenated_unique: all tests passed\n";
    return 0;
}
