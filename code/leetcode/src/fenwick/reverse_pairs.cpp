// 493. 翻转对
// 见 reverse_pairs.py 的题目与思路说明。
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

int reversePairs(std::vector<int> nums) {
    std::vector<long long> vals;
    vals.reserve(nums.size() * 2);
    for (int x : nums) {
        vals.push_back(x);
        vals.push_back(2LL * x);
    }
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

    Fenwick bit(static_cast<int>(vals.size()));
    long long ans = 0;
    int processed = 0;
    for (int x : nums) {
        long long twox = 2LL * x;
        int r = static_cast<int>(
                    std::lower_bound(vals.begin(), vals.end(), twox) - vals.begin()) +
                1;
        ans += processed - bit.prefix(r);

        int rx = static_cast<int>(
                     std::lower_bound(vals.begin(), vals.end(), static_cast<long long>(x)) -
                     vals.begin()) +
                 1;
        bit.add(rx, 1);
        ++processed;
    }
    return static_cast<int>(ans);
}

int main() {
    assert(reversePairs({1, 3, 2, 3, 1}) == 2);
    assert(reversePairs({2, 4, 3, 5, 1}) == 3);
    assert(reversePairs({1, 2, 3, 4}) == 0);
    assert(reversePairs({-5, -5}) == 1);
    assert(reversePairs({5, -5}) == 1);
    assert(reversePairs({}) == 0);

    std::cout << "reverse_pairs: all tests passed\n";
    return 0;
}
