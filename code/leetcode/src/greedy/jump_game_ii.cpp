// 45. 跳跃游戏 II
// 见 jump_game_ii.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int jump(const std::vector<int> &nums) {
    int n = static_cast<int>(nums.size());
    if (n <= 1) return 0;
    int jumps = 0;
    int curEnd = 0;
    int farthest = 0;
    for (int i = 0; i < n - 1; ++i) {
        farthest = std::max(farthest, i + nums[i]);
        if (i == curEnd) {
            ++jumps;
            curEnd = farthest;
        }
    }
    return jumps;
}

int main() {
    std::vector<int> a = {2, 3, 1, 1, 4};
    assert(jump(a) == 2);
    std::vector<int> b = {2, 3, 0, 1, 4};
    assert(jump(b) == 2);
    std::vector<int> c = {0};
    assert(jump(c) == 0);
    std::vector<int> d = {1, 2, 3};
    assert(jump(d) == 2);
    std::vector<int> e = {2, 0, 1, 1, 4};
    assert(jump(e) == 3);
    std::cout << "jump_game_ii: all tests passed\n";
    return 0;
}
