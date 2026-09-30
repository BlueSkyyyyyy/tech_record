// 283. 移动零（快慢指针 + 原地交换）
// 见 move_zeroes.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>

void moveZeroes(std::vector<int> &nums) {
    int slow = 0;
    for (int fast = 0; fast < static_cast<int>(nums.size()); ++fast) {
        if (nums[fast] != 0) {
            std::swap(nums[slow], nums[fast]);
            ++slow;
        }
    }
}

int main() {
    {
        std::vector<int> a{0, 1, 0, 3, 12};
        moveZeroes(a);
        assert((a == std::vector<int>{1, 3, 12, 0, 0}));
    }
    {
        std::vector<int> a{0};
        moveZeroes(a);
        assert((a == std::vector<int>{0}));
    }
    {
        std::vector<int> a{1, 2, 3};
        moveZeroes(a);
        assert((a == std::vector<int>{1, 2, 3}));
    }
    {
        std::vector<int> a{0, 0, 0, 1};
        moveZeroes(a);
        assert((a == std::vector<int>{1, 0, 0, 0}));
    }
    std::cout << "move_zeroes: all tests passed\n";
    return 0;
}
