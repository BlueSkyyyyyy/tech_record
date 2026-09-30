// 88. 合并两个有序数组（从后向前的三指针）
// 见 merge_sorted_array.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

void merge(std::vector<int> &nums1, int m, std::vector<int> &nums2, int n) {
    int i = m - 1, j = n - 1, k = m + n - 1;
    while (j >= 0) {
        if (i >= 0 && nums1[i] > nums2[j]) {
            nums1[k] = nums1[i];
            --i;
        } else {
            nums1[k] = nums2[j];
            --j;
        }
        --k;
    }
}

int main() {
    {
        std::vector<int> a{1, 2, 3, 0, 0, 0};
        std::vector<int> b{2, 5, 6};
        merge(a, 3, b, 3);
        std::vector<int> want{1, 2, 2, 3, 5, 6};
        assert(a == want);
    }
    {
        std::vector<int> a{1};
        std::vector<int> b{};
        merge(a, 1, b, 0);
        std::vector<int> want{1};
        assert(a == want);
    }
    {
        std::vector<int> a{0};
        std::vector<int> b{1};
        merge(a, 0, b, 1);
        std::vector<int> want{1};
        assert(a == want);
    }
    {
        std::vector<int> a{4, 5, 6, 0, 0, 0};
        std::vector<int> b{1, 2, 3};
        merge(a, 3, b, 3);
        std::vector<int> want{1, 2, 3, 4, 5, 6};
        assert(a == want);
    }
    std::cout << "merge_sorted_array: all tests passed\n";
    return 0;
}
