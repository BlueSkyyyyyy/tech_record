// 136. 只出现一次的数字
// 见 single_number.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int singleNumber(const std::vector<int> &nums) {
    int result = 0;
    for (int x : nums) {
        result ^= x;
    }
    return result;
}

int main() {
    assert(singleNumber({2, 2, 1}) == 1);
    assert(singleNumber({4, 1, 2, 1, 2}) == 4);
    assert(singleNumber({1}) == 1);
    assert(singleNumber({-1, -1, -2}) == -2);
    assert(singleNumber({0, 0, 5}) == 5);
    assert(singleNumber({7, 3, 7}) == 3);

    std::cout << "single_number: all tests passed\n";
    return 0;
}
