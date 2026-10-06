// 307. 区域和检索 - 数组可修改
// 见 range_sum_query_mutable.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

class NumArray {
    int n;
    std::vector<long long> tree;

public:
    explicit NumArray(std::vector<int> nums) : n(static_cast<int>(nums.size())), tree(2 * n, 0) {
        for (int i = 0; i < n; ++i) tree[n + i] = nums[i];
        for (int i = n - 1; i > 0; --i) tree[i] = tree[2 * i] + tree[2 * i + 1];
    }

    void update(int index, int val) {
        int i = index + n;
        tree[i] = val;
        for (i /= 2; i > 0; i /= 2) tree[i] = tree[2 * i] + tree[2 * i + 1];
    }

    long long sumRange(int left, int right) {
        long long res = 0;
        for (int l = left + n, r = right + n + 1; l < r; l /= 2, r /= 2) {
            if (l & 1) res += tree[l++];
            if (r & 1) res += tree[--r];
        }
        return res;
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

    NumArray c(std::vector<int>{-1});
    assert(c.sumRange(0, 0) == -1);
    c.update(0, 7);
    assert(c.sumRange(0, 0) == 7);

    std::cout << "range_sum_query_mutable: all tests passed\n";
    return 0;
}
