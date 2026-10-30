// 887. 鸡蛋掉落（DP + 二分优化转移）
// 见 super_egg_drop.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>
#include <vector>

int superEggDrop(int k, int n) {
    // dp[eggs][floors]：eggs 枚蛋、确定 floors 层需要的最少次数
    const int INF = 1000000000;
    std::vector<std::vector<int>> dp(k + 1, std::vector<int>(n + 1, 0));
    for (int floors = 1; floors <= n; ++floors) dp[0][floors] = INF;

    for (int eggs = 1; eggs <= k; ++eggs) {
        for (int floors = 1; floors <= n; ++floors) {
            int lo = 1, hi = floors, best = INF;
            while (lo <= hi) {
                int mid = lo + (hi - lo) / 2;
                int broken = dp[eggs - 1][mid - 1];
                int intact = dp[eggs][floors - mid];
                int worst = 1 + std::max(broken, intact);
                if (worst < best) best = worst;
                if (broken > intact) {
                    hi = mid - 1;
                } else {
                    lo = mid + 1;
                }
            }
            dp[eggs][floors] = best;
        }
    }
    return dp[k][n];
}

int main() {
    assert(superEggDrop(1, 2) == 2);
    assert(superEggDrop(2, 6) == 3);
    assert(superEggDrop(3, 14) == 4);
    assert(superEggDrop(2, 1) == 1);
    assert(superEggDrop(1, 10) == 10);
    std::cout << "super_egg_drop: all tests passed\n";
    return 0;
}
