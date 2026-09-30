// 454. 四数相加 II（两两之和 + 哈希计数）
// 见 four_sum_ii.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

int fourSumCount(const std::vector<int> &a, const std::vector<int> &b,
                 const std::vector<int> &c, const std::vector<int> &d) {
    std::unordered_map<int, int> ab;
    for (int x : a)
        for (int y : b) ++ab[x + y];
    int total = 0;
    for (int x : c)
        for (int y : d) {
            auto it = ab.find(-(x + y));
            if (it != ab.end()) total += it->second;
        }
    return total;
}

int main() {
    assert(fourSumCount({1, 2}, {-2, -1}, {-1, 2}, {0, 2}) == 2);
    assert(fourSumCount({0}, {0}, {0}, {0}) == 1);
    assert(fourSumCount({1}, {1}, {1}, {1}) == 0);
    std::cout << "four_sum_ii: all tests passed\n";
    return 0;
}
