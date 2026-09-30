// 26. 删除有序数组中的重复项（快慢指针）
// 见 remove_duplicates_sorted.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int removeDuplicates(std::vector<int> &nums) {
    if (nums.empty()) return 0;
    int slow = 0;
    for (int fast = 1; fast < static_cast<int>(nums.size()); ++fast) {
        if (nums[fast] != nums[slow]) {
            ++slow;
            nums[slow] = nums[fast];
        }
    }
    return slow + 1;
}

int main() {
    {
        std::vector<int> a{1, 1, 2};
        int k = removeDuplicates(a);
        std::vector<int> got(a.begin(), a.begin() + k);
        std::vector<int> want{1, 2};
        assert(k == 2 && got == want);
    }
    {
        std::vector<int> a{0, 0, 1, 1, 1, 2, 2, 3, 3, 4};
        int k = removeDuplicates(a);
        std::vector<int> got(a.begin(), a.begin() + k);
        std::vector<int> want{0, 1, 2, 3, 4};
        assert(k == 5 && got == want);
    }
    {
        std::vector<int> a{};
        assert(removeDuplicates(a) == 0);
    }
    {
        std::vector<int> a{7};
        assert(removeDuplicates(a) == 1);
    }
    std::cout << "remove_duplicates_sorted: all tests passed\n";
    return 0;
}
