// 1434. 每个人戴不同帽子的方案数
// 见 number_ways_wear_hats.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <set>
#include <vector>

int numberWays(std::vector<std::vector<int>> &hats) {
    const int MOD = 1000000007;
    int n = static_cast<int>(hats.size());

    std::vector<std::vector<int>> likers(41);
    for (int i = 0; i < n; ++i) {
        std::set<int> uniq(hats[i].begin(), hats[i].end());
        for (int h : uniq) {
            likers[h].push_back(i);
        }
    }

    std::vector<long long> dp(1 << n, 0);
    dp[0] = 1;
    for (int h = 1; h <= 40; ++h) {
        if (likers[h].empty()) {
            continue;
        }
        std::vector<long long> nxt = dp;
        for (int mask = 0; mask < (1 << n); ++mask) {
            if (dp[mask] == 0) {
                continue;
            }
            for (int i : likers[h]) {
                if (!(mask >> i & 1)) {
                    nxt[mask | (1 << i)] = (nxt[mask | (1 << i)] + dp[mask]) % MOD;
                }
            }
        }
        dp = nxt;
    }
    return static_cast<int>(dp[(1 << n) - 1]);
}

int main() {
    {
        std::vector<std::vector<int>> hats = {{3, 4}, {4, 5}, {5}};
        assert(numberWays(hats) == 1);
    }
    {
        std::vector<std::vector<int>> hats = {{3, 5, 1}, {3, 5}};
        assert(numberWays(hats) == 4);
    }
    {
        std::vector<std::vector<int>> hats = {{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}};
        assert(numberWays(hats) == 24);
    }
    {
        std::vector<std::vector<int>> hats = {{1}, {1}};
        assert(numberWays(hats) == 0);
    }

    std::cout << "number_ways_wear_hats: all tests passed\n";
    return 0;
}
