// 303. 区域和检索 - 数组不可变（前缀和）
// 见 range_sum_query.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

class NumArray {
public:
    explicit NumArray(const std::vector<int> &nums) {
        prefix_.resize(nums.size() + 1, 0);
        for (std::size_t i = 0; i < nums.size(); ++i)
            prefix_[i + 1] = prefix_[i] + nums[i];
    }

    int sumRange(int left, int right) const {
        return prefix_[right + 1] - prefix_[left];
    }

private:
    std::vector<int> prefix_;
};

int main() {
    NumArray arr({-2, 0, 3, -5, 2, -1});
    assert(arr.sumRange(0, 2) == 1);
    assert(arr.sumRange(2, 5) == -1);
    assert(arr.sumRange(0, 5) == -3);
    assert(arr.sumRange(3, 3) == -5);
    NumArray arr2({5});
    assert(arr2.sumRange(0, 0) == 5);
    std::cout << "range_sum_query: all tests passed\n";
    return 0;
}
