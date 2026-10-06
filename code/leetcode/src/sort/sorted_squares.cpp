// 977. 有序数组的平方
// 见 sorted_squares.py 的题目与思路说明。
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

std::vector<int> sortedSquares(const std::vector<int> &nums) {
    int n = static_cast<int>(nums.size());
    std::vector<int> res(n, 0);
    int l = 0, r = n - 1, k = n - 1;
    while (l <= r) {
        if (std::abs(nums[l]) > std::abs(nums[r])) {
            res[k] = nums[l] * nums[l];
            ++l;
        } else {
            res[k] = nums[r] * nums[r];
            --r;
        }
        --k;
    }
    return res;
}

int main() {
    std::vector<int> want1 = {0, 1, 9, 16, 100};
    assert(sortedSquares({-4, -1, 0, 3, 10}) == want1);

    std::vector<int> want2 = {4, 9, 9, 49, 121};
    assert(sortedSquares({-7, -3, 2, 3, 11}) == want2);

    std::vector<int> want3 = {0};
    assert(sortedSquares({0}) == want3);

    std::vector<int> want4 = {1, 4, 9};
    assert(sortedSquares({-3, -2, -1}) == want4);

    std::vector<int> want5 = {1, 4, 9};
    assert(sortedSquares({1, 2, 3}) == want5);

    std::cout << "sorted_squares: all tests passed\n";
    return 0;
}
