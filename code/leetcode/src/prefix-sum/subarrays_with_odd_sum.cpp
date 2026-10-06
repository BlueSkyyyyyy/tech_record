// 1524. 和为奇数的子数组数目（同余前缀和 mod 2 的特例）
// 见 subarrays_with_odd_sum.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

constexpr long long MOD = 1000000007LL;

long long numOfSubarraysWithOddSum(const std::vector<int> &arr) {
    long long even = 1;
    long long odd = 0;
    long long prefix = 0;
    for (int x : arr) {
        prefix += x;
        if (prefix % 2 != 0)
            ++odd;
        else
            ++even;
    }
    return (even * odd) % MOD;
}

int main() {
    assert(numOfSubarraysWithOddSum({1, 3, 5}) == 4);
    assert(numOfSubarraysWithOddSum({2, 4, 6}) == 0);
    assert(numOfSubarraysWithOddSum({1, 2, 3, 4, 5, 6, 7}) == 16);
    assert(numOfSubarraysWithOddSum({1}) == 1);
    assert(numOfSubarraysWithOddSum({100}) == 0);
    assert(numOfSubarraysWithOddSum({1, 1}) == 2);
    assert(numOfSubarraysWithOddSum({1, 2}) == 2);
    std::cout << "subarrays_with_odd_sum: all tests passed\n";
    return 0;
}
