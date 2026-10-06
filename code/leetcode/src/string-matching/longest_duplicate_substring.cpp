// 1044. 最长重复子串
// 见 longest_duplicate_substring.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

const long long MOD = 1000000007LL;
const long long BASE = 131LL;

bool check(int length, const std::string &s, const std::vector<long long> &pw,
           std::string &out) {
    if (length == 0) {
        out = "";
        return true;
    }
    int n = static_cast<int>(s.size());
    long long power = pw[length - 1];
    long long h = 0;
    for (int i = 0; i < length; ++i) {
        h = (h * BASE + static_cast<unsigned char>(s[i])) % MOD;
    }
    std::unordered_map<long long, int> seen;
    seen[h] = 0;
    for (int i = 1; i + length <= n; ++i) {
        h = ((h - static_cast<unsigned char>(s[i - 1]) * power) % MOD + MOD) % MOD;
        h = (h * BASE + static_cast<unsigned char>(s[i + length - 1])) % MOD;
        auto it = seen.find(h);
        if (it != seen.end() &&
            s.compare(it->second, length, s, i, length) == 0) {
            out = s.substr(i, length);
            return true;
        }
        seen[h] = i;
    }
    return false;
}

std::string longestDupSubstring(const std::string &s) {
    int n = static_cast<int>(s.size());
    if (n < 2) {
        return "";
    }
    std::vector<long long> pw(n, 1);
    for (int i = 1; i < n; ++i) {
        pw[i] = pw[i - 1] * BASE % MOD;
    }
    int lo = 0, hi = n - 1;
    std::string ans;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        std::string out;
        if (check(mid, s, pw, out)) {
            ans = out;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return ans;
}

int main() {
    assert(longestDupSubstring("banana") == "ana");
    assert(longestDupSubstring("abcd") == "");
    assert(longestDupSubstring("aa") == "a");
    assert(longestDupSubstring("aaaa") == "aaa");
    assert(longestDupSubstring("a") == "");
    assert(longestDupSubstring("abcabcabcd") == "abcabc");

    std::cout << "longestDupSubstring: all tests passed\n";
    return 0;
}
