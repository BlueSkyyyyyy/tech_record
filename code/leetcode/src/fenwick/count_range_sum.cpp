// 327. 区间和的个数
// 见 count_range_sum.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
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

int countRangeSum(std::vector<int> nums, int lower, int upper) {
    int n = static_cast<int>(nums.size());
    std::vector<long long> prefix(n + 1, 0);
    for (int i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + nums[i];

    std::vector<long long> vals = prefix;
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

    auto rank = [&](long long v) {
        return static_cast<int>(std::lower_bound(vals.begin(), vals.end(), v) -
                                vals.begin()) +
               1;
    };

    Fenwick bit(static_cast<int>(vals.size()));
    long long ans = 0;
    bit.add(rank(prefix[0]), 1);
    for (int j = 1; j <= n; ++j) {
        long long lo = prefix[j] - upper;
        long long hi = prefix[j] - lower;
        int left = static_cast<int>(std::lower_bound(vals.begin(), vals.end(), lo) -
                                    vals.begin());
        int right = static_cast<int>(std::upper_bound(vals.begin(), vals.end(), hi) -
                                     vals.begin());
        ans += bit.prefix(right) - bit.prefix(left);
        bit.add(rank(prefix[j]), 1);
    }
    return static_cast<int>(ans);
}

int main() {
    assert(countRangeSum({-2, 5, -1}, -2, 2) == 3);
    assert(countRangeSum({0}, 0, 0) == 1);
    assert(countRangeSum({0}, 1, 1) == 0);
    assert(countRangeSum({0, 0, 0}, 0, 0) == 6);
    assert(countRangeSum({-1, -1, -1}, -1, 0) == 3);
    assert(countRangeSum({1, 2, 3}, 3, 6) == 4);

    std::cout << "count_range_sum: all tests passed\n";
    return 0;
}
