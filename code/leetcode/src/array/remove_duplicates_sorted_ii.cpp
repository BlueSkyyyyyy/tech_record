// 80. 删除有序数组中的重复项 II（快慢指针 + 回头看两位）
// 见 remove_duplicates_sorted_ii.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int removeDuplicatesII(std::vector<int> &nums) {
    int slow = 0;
    for (int fast = 0; fast < static_cast<int>(nums.size()); ++fast) {
        if (slow < 2 || nums[fast] != nums[slow - 2]) {
            nums[slow] = nums[fast];
            ++slow;
        }
    }
    return slow;
}

int main() {
    {
        std::vector<int> a{1, 1, 1, 2, 2, 3};
        int k = removeDuplicatesII(a);
        std::vector<int> got(a.begin(), a.begin() + k);
        std::vector<int> want{1, 1, 2, 2, 3};
        assert(k == 5 && got == want);
    }
    {
        std::vector<int> a{0, 0, 1, 1, 1, 1, 2, 3, 3};
        int k = removeDuplicatesII(a);
        std::vector<int> got(a.begin(), a.begin() + k);
        std::vector<int> want{0, 0, 1, 1, 2, 3, 3};
        assert(k == 7 && got == want);
    }
    {
        std::vector<int> a{};
        assert(removeDuplicatesII(a) == 0);
    }
    {
        std::vector<int> a{1, 1, 1};
        int k = removeDuplicatesII(a);
        std::vector<int> got(a.begin(), a.begin() + k);
        std::vector<int> want{1, 1};
        assert(k == 2 && got == want);
    }
    std::cout << "remove_duplicates_sorted_ii: all tests passed\n";
    return 0;
}
