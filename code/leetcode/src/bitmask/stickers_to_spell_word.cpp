// 691. 贴纸拼词
// 见 stickers_to_spell_word.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

std::vector<std::string> gStickers;
std::string gTarget;
std::unordered_map<int, int> gMemo;

int solve(int remaining) {
    if (remaining == 0) {
        return 0;
    }
    auto it = gMemo.find(remaining);
    if (it != gMemo.end()) {
        return it->second;
    }
    int T = static_cast<int>(gTarget.size());
    int best = 1e9;
    for (const std::string &sticker : gStickers) {
        int cnt[26] = {0};
        for (char ch : sticker) {
            cnt[ch - 'a']++;
        }
        int nxt = remaining;
        for (int i = 0; i < T; ++i) {
            if (nxt >> i & 1) {
                int idx = gTarget[i] - 'a';
                if (cnt[idx] > 0) {
                    cnt[idx]--;
                    nxt ^= 1 << i;
                }
            }
        }
        if (nxt != remaining) {
            best = std::min(best, 1 + solve(nxt));
        }
    }
    return gMemo[remaining] = best;
}

int minStickers(std::vector<std::string> &stickers, std::string target) {
    gStickers = stickers;
    gTarget = target;
    gMemo.clear();
    int T = static_cast<int>(target.size());
    int ans = solve((1 << T) - 1);
    return ans >= 1e9 ? -1 : ans;
}

int main() {
    {
        std::vector<std::string> s = {"with", "example", "science"};
        assert(minStickers(s, "thehat") == 3);
    }
    {
        std::vector<std::string> s = {"notice", "possible"};
        assert(minStickers(s, "basicbasic") == -1);
    }
    {
        std::vector<std::string> s = {"a"};
        assert(minStickers(s, "a") == 1);
    }
    {
        std::vector<std::string> s = {"ab", "bc", "cd"};
        assert(minStickers(s, "abcd") == 2);
    }
    std::cout << "stickers_to_spell_word: all tests passed\n";
    return 0;
}
