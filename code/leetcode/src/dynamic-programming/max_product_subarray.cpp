// 152. 乘积最大子数组
// 见 max_product_subarray.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

long long maxProduct(const std::vector<int> &nums) {
    long long curMax = nums[0], curMin = nums[0], best = nums[0];
    for (size_t i = 1; i < nums.size(); ++i) {
        long long x = nums[i];
        long long a = x, b = curMax * x, c = curMin * x;
        long long newMax = std::max({a, b, c});
        long long newMin = std::min({a, b, c});
        curMax = newMax;
        curMin = newMin;
        best = std::max(best, curMax);
    }
    return best;
}

int main() {
    std::vector<int> a = {2, 3, -2, 4};
    std::vector<int> b = {-2, 0, -1};
    std::vector<int> c = {-2, 3, -4};
    std::vector<int> d = {0, 2};
    std::vector<int> e = {-2};
    std::vector<int> f = {1, -2, 3, -4};
    assert(maxProduct(a) == 6);
    assert(maxProduct(b) == 0);
    assert(maxProduct(c) == 24);
    assert(maxProduct(d) == 2);
    assert(maxProduct(e) == -2);
    assert(maxProduct(f) == 24);
    std::cout << "max_product_subarray: all tests passed\n";
    return 0;
}
