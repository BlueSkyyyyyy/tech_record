// 1049. 最后一块石头的重量 II
// 见 last_stone_weight_ii.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int lastStoneWeightII(const std::vector<int> &stones) {
    int total = 0;
    for (int w : stones) total += w;
    int target = total / 2;
    std::vector<bool> dp(target + 1, false);
    dp[0] = true;
    for (int w : stones) {
        for (int i = target; i >= w; --i) {
            dp[i] = dp[i] || dp[i - w];
        }
    }
    for (int s = target; s >= 0; --s) {
        if (dp[s]) return total - 2 * s;
    }
    return total;
}

int main() {
    std::vector<int> a = {2, 7, 4, 1, 8, 1};
    std::vector<int> b = {31, 26, 33, 21, 40};
    std::vector<int> c = {1, 2};
    std::vector<int> d = {1};
    std::vector<int> e = {1, 1, 1};
    std::vector<int> f = {2, 2};
    assert(lastStoneWeightII(a) == 1);
    assert(lastStoneWeightII(b) == 5);
    assert(lastStoneWeightII(c) == 1);
    assert(lastStoneWeightII(d) == 1);
    assert(lastStoneWeightII(e) == 1);
    assert(lastStoneWeightII(f) == 0);
    std::cout << "last_stone_weight_ii: all tests passed\n";
    return 0;
}
