// 974. 和可被 K 整除的子数组（同余前缀和 + 哈希）
// 见 subarrays_divisible_by_k.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

long long subarraysDivisibleByK(const std::vector<int> &nums, int k) {
    std::unordered_map<int, long long> count;
    count[0] = 1;
    long long prefix = 0;
    long long res = 0;
    for (int x : nums) {
        prefix = ((prefix + x) % k + k) % k;
        res += count[prefix];
        ++count[prefix];
    }
    return res;
}

int main() {
    assert(subarraysDivisibleByK({4, 5, 0, -2, -3, 1}, 5) == 7);
    assert(subarraysDivisibleByK({5}, 9) == 0);
    assert(subarraysDivisibleByK({-1, 2, 9}, 2) == 2);
    assert(subarraysDivisibleByK({0, 0}, 1) == 3);
    assert(subarraysDivisibleByK({1, 2, 3}, 3) == 3);
    std::cout << "subarrays_divisible_by_k: all tests passed\n";
    return 0;
}
