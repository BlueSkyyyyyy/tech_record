// 78. 子集 · 位枚举法
// 见 subsets_bitmask.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

std::vector<std::vector<int>> subsets(const std::vector<int> &nums) {
    int n = static_cast<int>(nums.size());
    std::vector<std::vector<int>> res;
    for (int mask = 0; mask < (1 << n); ++mask) {
        std::vector<int> subset;
        for (int i = 0; i < n; ++i) {
            if (mask & (1 << i)) {
                subset.push_back(nums[i]);
            }
        }
        res.push_back(subset);
    }
    return res;
}

int main() {
    auto got = subsets({1, 2, 3});
    std::vector<std::vector<int>> want = {
        {}, {1}, {2}, {1, 2}, {3}, {1, 3}, {2, 3}, {1, 2, 3}};
    assert(got.size() == 8);
    for (int i = 0; i < 8; ++i) {
        auto a = got[i];
        auto b = want[i];
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        assert(a == b);
    }
    assert(subsets({}).size() == 1);
    assert(subsets({0}).size() == 2);

    std::cout << "subsets_bitmask: all tests passed\n";
    return 0;
}
