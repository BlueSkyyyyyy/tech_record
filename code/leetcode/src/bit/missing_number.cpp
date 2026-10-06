// 268. 丢失的数字
// 见 missing_number.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int missingNumber(const std::vector<int> &nums) {
    int result = static_cast<int>(nums.size());
    for (std::size_t i = 0; i < nums.size(); ++i) {
        result ^= static_cast<int>(i) ^ nums[i];
    }
    return result;
}

int main() {
    assert(missingNumber({3, 0, 1}) == 2);
    assert(missingNumber({0, 1}) == 2);
    assert(missingNumber({9, 6, 4, 2, 3, 5, 7, 0, 1}) == 8);
    assert(missingNumber({0}) == 1);
    assert(missingNumber({1}) == 0);

    std::cout << "missing_number: all tests passed\n";
    return 0;
}
