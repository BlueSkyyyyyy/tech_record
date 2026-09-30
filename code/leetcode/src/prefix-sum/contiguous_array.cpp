// 525. 连续数组（前缀和 + 最早下标）
// 见 contiguous_array.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

int findMaxLength(const std::vector<int> &nums) {
    std::unordered_map<int, int> first;
    first[0] = -1;
    int count = 0;
    int res = 0;
    for (int i = 0; i < static_cast<int>(nums.size()); ++i) {
        count += (nums[i] == 1) ? 1 : -1;
        auto it = first.find(count);
        if (it != first.end()) {
            if (i - it->second > res) res = i - it->second;
        } else {
            first[count] = i;
        }
    }
    return res;
}

int main() {
    assert(findMaxLength({0, 1}) == 2);
    assert(findMaxLength({0, 1, 0}) == 2);
    assert(findMaxLength({0, 0, 0, 1, 1, 1}) == 6);
    assert(findMaxLength({1, 1}) == 0);
    assert(findMaxLength({0, 1, 1, 0, 1, 0}) == 6);
    std::cout << "contiguous_array: all tests passed\n";
    return 0;
}
