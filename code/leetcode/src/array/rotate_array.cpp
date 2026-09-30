// 189. 轮转数组（三次反转）
// 见 rotate_array.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

void rotate(std::vector<int> &nums, int k) {
    int n = static_cast<int>(nums.size());
    if (n == 0) return;
    k %= n;
    std::reverse(nums.begin(), nums.end());
    std::reverse(nums.begin(), nums.begin() + k);
    std::reverse(nums.begin() + k, nums.end());
}

int main() {
    {
        std::vector<int> a{1, 2, 3, 4, 5, 6, 7};
        rotate(a, 3);
        std::vector<int> want{5, 6, 7, 1, 2, 3, 4};
        assert(a == want);
    }
    {
        std::vector<int> a{-1, -100, 3, 99};
        rotate(a, 2);
        std::vector<int> want{3, 99, -1, -100};
        assert(a == want);
    }
    {
        std::vector<int> a{1, 2};
        rotate(a, 0);
        std::vector<int> want{1, 2};
        assert(a == want);
    }
    {
        std::vector<int> a{1, 2};
        rotate(a, 4);
        std::vector<int> want{1, 2};
        assert(a == want);
    }
    {
        std::vector<int> a{1};
        rotate(a, 1);
        std::vector<int> want{1};
        assert(a == want);
    }
    std::cout << "rotate_array: all tests passed\n";
    return 0;
}
