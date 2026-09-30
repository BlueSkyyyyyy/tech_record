// 238. 除自身以外数组的乘积（前缀积 / 左右乘积）
// 见 product_of_array_except_self.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

std::vector<long long> productExceptSelf(const std::vector<int> &nums) {
    int n = static_cast<int>(nums.size());
    std::vector<long long> res(n, 1);
    long long left = 1;
    for (int i = 0; i < n; ++i) {
        res[i] = left;
        left *= nums[i];
    }
    long long right = 1;
    for (int i = n - 1; i >= 0; --i) {
        res[i] *= right;
        right *= nums[i];
    }
    return res;
}

int main() {
    std::vector<long long> got1 = productExceptSelf({1, 2, 3, 4});
    std::vector<long long> want1 = {24, 12, 8, 6};
    assert(got1 == want1);
    std::vector<long long> got2 = productExceptSelf({-1, 1, 0, -3, 3});
    std::vector<long long> want2 = {0, 0, 9, 0, 0};
    assert(got2 == want2);
    std::vector<long long> got3 = productExceptSelf({2, 3});
    std::vector<long long> want3 = {3, 2};
    assert(got3 == want3);
    std::vector<long long> got4 = productExceptSelf({0, 0});
    std::vector<long long> want4 = {0, 0};
    assert(got4 == want4);
    std::vector<long long> got5 = productExceptSelf({5});
    std::vector<long long> want5 = {1};
    assert(got5 == want5);
    std::cout << "product_of_array_except_self: all tests passed\n";
    return 0;
}
