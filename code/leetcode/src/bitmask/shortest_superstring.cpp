// 943. 最短超级串
// 见 shortest_superstring.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

std::string shortestSuperstring(std::vector<std::string> words) {
    std::vector<std::string> unique;
    for (const std::string &w : words) {
        if (std::find(unique.begin(), unique.end(), w) == unique.end()) {
            unique.push_back(w);
        }
    }
    words.clear();
    for (const std::string &w : unique) {
        bool redundant = false;
        for (const std::string &other : unique) {
            if (w != other && other.find(w) != std::string::npos) {
                redundant = true;
                break;
            }
        }
        if (!redundant) {
            words.push_back(w);
        }
    }

    int k = static_cast<int>(words.size());
    if (k == 0) {
        return "";
    }
    std::vector<std::vector<int>> overlap(k, std::vector<int>(k, 0));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            if (i == j) {
                continue;
            }
            const std::string &a = words[i];
            const std::string &b = words[j];
            int limit = std::min(a.size(), b.size());
            for (int length = limit; length > 0; --length) {
                if (a.substr(a.size() - length) == b.substr(0, length)) {
                    overlap[i][j] = length;
                    break;
                }
            }
        }
    }

    int full = (1 << k) - 1;
    const int INF = 1e9;
    std::vector<std::vector<int>> dp(1 << k, std::vector<int>(k, INF));
    std::vector<std::vector<int>> parent(1 << k, std::vector<int>(k, -1));
    for (int i = 0; i < k; ++i) {
        dp[1 << i][i] = static_cast<int>(words[i].size());
    }

    for (int mask = 0; mask < (1 << k); ++mask) {
        for (int last = 0; last < k; ++last) {
            if (dp[mask][last] == INF) {
                continue;
            }
            for (int nxt = 0; nxt < k; ++nxt) {
                if (mask >> nxt & 1) {
                    continue;
                }
                int nm = mask | (1 << nxt);
                int cand = dp[mask][last] + static_cast<int>(words[nxt].size()) - overlap[last][nxt];
                if (cand < dp[nm][nxt]) {
                    dp[nm][nxt] = cand;
                    parent[nm][nxt] = last;
                }
            }
        }
    }

    int bestLen = INF;
    int bestLast = -1;
    for (int i = 0; i < k; ++i) {
        if (dp[full][i] < bestLen) {
            bestLen = dp[full][i];
            bestLast = i;
        }
    }

    std::vector<int> order;
    int mask = full;
    int last = bestLast;
    while (last != -1) {
        order.push_back(last);
        int prev = parent[mask][last];
        mask ^= 1 << last;
        last = prev;
    }
    std::reverse(order.begin(), order.end());

    std::string ans = words[order[0]];
    for (int t = 1; t < static_cast<int>(order.size()); ++t) {
        int i = order[t - 1];
        int j = order[t];
        ans += words[j].substr(overlap[i][j]);
    }
    return ans;
}

static bool containsAll(const std::string &s, const std::vector<std::string> &ws) {
    for (const std::string &w : ws) {
        if (s.find(w) == std::string::npos) {
            return false;
        }
    }
    return true;
}

int main() {
    {
        std::vector<std::string> ws = {"alex", "loves", "leetcode"};
        auto got = shortestSuperstring(ws);
        assert(containsAll(got, ws));
        assert(got.size() == 17);
    }
    {
        std::vector<std::string> ws = {"catg", "ctaagt", "gcta", "ttca", "atgcatc"};
        auto got = shortestSuperstring(ws);
        assert(containsAll(got, ws));
        assert(got.size() == 16);
    }
    assert(shortestSuperstring({"abc"}) == "abc");
    assert(shortestSuperstring({"abc", "bcd"}) == "abcd");

    std::cout << "shortest_superstring: all tests passed\n";
    return 0;
}
