// 674. 最长连续递增序列
// 见 longest_continuous_increasing_subsequence.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

int findLengthOfLCIS(const std::vector<int> &nums) {
    if (nums.empty()) return 0;
    int best = 1, cur = 1;
    for (size_t i = 1; i < nums.size(); ++i) {
        if (nums[i] > nums[i - 1]) {
            ++cur;
        } else {
            cur = 1;
        }
        best = std::max(best, cur);
    }
    return best;
}

int main() {
    std::vector<int> a = {1, 3, 5, 4, 7};
    std::vector<int> b = {2, 2, 2, 2, 2};
    std::vector<int> c = {1, 3, 5, 7};
    std::vector<int> d = {};
    std::vector<int> e = {1};
    assert(findLengthOfLCIS(a) == 3);
    assert(findLengthOfLCIS(b) == 1);
    assert(findLengthOfLCIS(c) == 4);
    assert(findLengthOfLCIS(d) == 0);
    assert(findLengthOfLCIS(e) == 1);
    std::cout << "longest_continuous_increasing_subsequence: all tests passed\n";
    return 0;
}
