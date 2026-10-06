// 55. 跳跃游戏
// 见 jump_game.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

bool canJump(const std::vector<int> &nums) {
    int reach = 0;
    for (int i = 0; i < static_cast<int>(nums.size()); ++i) {
        if (i > reach) return false;
        reach = std::max(reach, i + nums[i]);
    }
    return true;
}

int main() {
    std::vector<int> a = {2, 3, 1, 1, 4};
    assert(canJump(a) == true);
    std::vector<int> b = {3, 2, 1, 0, 4};
    assert(canJump(b) == false);
    std::vector<int> c = {0};
    assert(canJump(c) == true);
    std::vector<int> d = {2, 0, 0};
    assert(canJump(d) == true);
    std::vector<int> e = {1, 0, 1};
    assert(canJump(e) == false);
    std::vector<int> f = {0, 1};
    assert(canJump(f) == false);
    std::cout << "jump_game: all tests passed\n";
    return 0;
}
