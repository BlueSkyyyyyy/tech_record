// 560. 和为 K 的子数组（前缀和 + 哈希计数）
// 见 subarray_sum_equals_k.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

int subarraySum(const std::vector<int> &nums, int k) {
    std::unordered_map<int, int> count{{0, 1}};
    int prefix = 0, total = 0;
    for (int x : nums) {
        prefix += x;
        auto it = count.find(prefix - k);
        if (it != count.end()) total += it->second;
        ++count[prefix];
    }
    return total;
}

int main() {
    assert(subarraySum({1, 1, 1}, 2) == 2);
    assert(subarraySum({1, 2, 3}, 3) == 2);
    assert(subarraySum({1, -1, 0}, 0) == 3);
    assert(subarraySum({-1, -1, 1}, 0) == 1);
    std::cout << "subarray_sum_equals_k: all tests passed\n";
    return 0;
}
