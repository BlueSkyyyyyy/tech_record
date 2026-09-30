// 128. 最长连续序列（哈希集合 + 只从起点扫描）
// 见 longest_consecutive_sequence.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_set>
#include <vector>

int longestConsecutive(const std::vector<int> &nums) {
    std::unordered_set<int> numSet(nums.begin(), nums.end());
    int best = 0;
    for (int x : numSet) {
        if (numSet.count(x - 1)) continue;
        int length = 1;
        while (numSet.count(x + length)) ++length;
        best = std::max(best, length);
    }
    return best;
}

int main() {
    assert(longestConsecutive({100, 4, 200, 1, 3, 2}) == 4);
    assert(longestConsecutive({0, 3, 7, 2, 5, 8, 4, 6, 0, 1}) == 9);
    assert(longestConsecutive({}) == 0);
    assert(longestConsecutive({1, 2, 0, 1}) == 3);
    std::cout << "longest_consecutive_sequence: all tests passed\n";
    return 0;
}
