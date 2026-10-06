// 307. 区域和检索 - 数组可修改
// 见 range_sum_query_mutable.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>

struct Fenwick {
    int n;
    std::vector<long long> tree;
    explicit Fenwick(int n) : n(n), tree(n + 1, 0) {}
    void add(int i, long long delta) {
        for (; i <= n; i += i & -i) tree[i] += delta;
    }
    long long prefix(int i) const {
        long long s = 0;
        for (; i > 0; i -= i & -i) s += tree[i];
        return s;
    }
};

class NumArray {
    std::vector<int> nums;
    Fenwick bit;

public:
    explicit NumArray(std::vector<int> nums_) : nums(std::move(nums_)), bit(nums.size()) {
        for (int i = 0; i < static_cast<int>(nums.size()); ++i) bit.add(i + 1, nums[i]);
    }

    void update(int index, int val) {
        bit.add(index + 1, static_cast<long long>(val) - nums[index]);
        nums[index] = val;
    }

    long long sumRange(int left, int right) {
        return bit.prefix(right + 1) - bit.prefix(left);
    }
};

int main() {
    NumArray a(std::vector<int>{1, 3, 5});
    assert(a.sumRange(0, 2) == 9);
    a.update(1, 2);
    assert(a.sumRange(0, 2) == 8);
    assert(a.sumRange(1, 1) == 2);

    NumArray b(std::vector<int>{0, 9, 5, 7, 3});
    assert(b.sumRange(4, 4) == 3);
    assert(b.sumRange(2, 4) == 15);
    b.update(4, 5);
    assert(b.sumRange(2, 4) == 17);
    b.update(0, -1);
    assert(b.sumRange(0, 4) == 25);

    std::cout << "range_sum_query_mutable: all tests passed\n";
    return 0;
}
