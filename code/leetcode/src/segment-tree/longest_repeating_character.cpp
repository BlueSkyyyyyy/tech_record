// 2213. 由单个字符重复的最长子字符串
// 见 longest_repeating_character.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

struct SegTree {
    int n;
    std::string s;
    std::vector<int> ln, lc, ll, rc, rl, bs;
    explicit SegTree(const std::string& str)
        : n(str.size()), s(str), ln(4 * n), lc(4 * n), ll(4 * n), rc(4 * n), rl(4 * n),
          bs(4 * n) {
        build(1, 0, n - 1);
    }
    void build(int o, int l, int r) {
        if (l == r) {
            ln[o] = 1;
            lc[o] = rc[o] = s[l];
            ll[o] = rl[o] = 1;
            bs[o] = 1;
            return;
        }
        int m = (l + r) / 2;
        build(2 * o, l, m);
        build(2 * o + 1, m + 1, r);
        pull(o);
    }
    void pull(int o) {
        int L = 2 * o, R = 2 * o + 1;
        ln[o] = ln[L] + ln[R];

        lc[o] = lc[L];
        ll[o] = ll[L];
        if (ll[L] == ln[L] && lc[L] == lc[R]) ll[o] = ln[L] + ll[R];

        rc[o] = rc[R];
        rl[o] = rl[R];
        if (rl[R] == ln[R] && rc[R] == rc[L]) rl[o] = ln[R] + rl[L];

        int best = std::max(bs[L], bs[R]);
        if (rc[L] == lc[R]) best = std::max(best, rl[L] + ll[R]);
        bs[o] = best;
    }
    void update(int o, int l, int r, int idx, char ch) {
        if (l == r) {
            lc[o] = rc[o] = ch;
            return;
        }
        int m = (l + r) / 2;
        if (idx <= m) update(2 * o, l, m, idx, ch);
        else update(2 * o + 1, m + 1, r, idx, ch);
        pull(o);
    }
    void update(int idx, char ch) { update(1, 0, n - 1, idx, ch); }
};

std::vector<int> longestRepeating(const std::string& s, const std::string& queryCharacters,
                                  const std::vector<int>& queryIndices) {
    SegTree st(s);
    std::vector<int> res;
    for (int i = 0; i < static_cast<int>(queryIndices.size()); ++i) {
        st.update(queryIndices[i], queryCharacters[i]);
        res.push_back(st.bs[1]);
    }
    return res;
}

int main() {
    assert(longestRepeating("babacc", "bcb", {1, 3, 3}) == (std::vector<int>{3, 3, 4}));
    assert(longestRepeating("abyzz", "aa", {2, 1}) == (std::vector<int>{2, 3}));
    assert(longestRepeating("a", "b", {0}) == (std::vector<int>{1}));
    std::cout << "longest_repeating_character: all tests passed\n";
    return 0;
}
