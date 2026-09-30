// 27. 移除元素（快慢指针）
// 见 remove_element.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int removeElement(std::vector<int> &nums, int val) {
    int slow = 0;
    for (int fast = 0; fast < static_cast<int>(nums.size()); ++fast) {
        if (nums[fast] != val) {
            nums[slow] = nums[fast];
            ++slow;
        }
    }
    return slow;
}

int main() {
    {
        std::vector<int> a{3, 2, 2, 3};
        int k = removeElement(a, 3);
        std::vector<int> got(a.begin(), a.begin() + k);
        std::sort(got.begin(), got.end());
        std::vector<int> want{2, 2};
        assert(k == 2 && got == want);
    }
    {
        std::vector<int> a{0, 1, 2, 2, 3, 0, 4, 2};
        int k = removeElement(a, 2);
        std::vector<int> got(a.begin(), a.begin() + k);
        std::sort(got.begin(), got.end());
        std::vector<int> want{0, 0, 1, 3, 4};
        assert(k == 5 && got == want);
    }
    {
        std::vector<int> a{};
        assert(removeElement(a, 1) == 0);
    }
    {
        std::vector<int> a{5, 5, 5};
        assert(removeElement(a, 5) == 0);
    }
    std::cout << "remove_element: all tests passed\n";
    return 0;
}
