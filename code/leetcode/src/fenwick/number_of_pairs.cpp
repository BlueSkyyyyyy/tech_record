// 2426. 满足不等式的数对数目
// 见 number_of_pairs.py 的题目与思路说明。
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

long long numberOfPairs(std::vector<int> nums1, std::vector<int> nums2, int k) {
    int n = static_cast<int>(nums1.size());
    std::vector<long long> diff(n);
    for (int i = 0; i < n; ++i)
        diff[i] = static_cast<long long>(nums1[i]) - nums2[i];

    std::vector<long long> vals = diff;
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

    Fenwick bit(static_cast<int>(vals.size()));
    long long ans = 0;
    for (int j = 0; j < n; ++j) {
        long long threshold = diff[j] + k;
        int idx = static_cast<int>(
            std::upper_bound(vals.begin(), vals.end(), threshold) - vals.begin());
        ans += bit.prefix(idx);

        int r = static_cast<int>(
                    std::lower_bound(vals.begin(), vals.end(), diff[j]) - vals.begin()) +
                1;
        bit.add(r, 1);
    }
    return ans;
}

int main() {
    assert(numberOfPairs({3, 2, 5}, {2, 2, 1}, 1) == 3);
    assert(numberOfPairs({3, -1}, {-2, 2}, -1) == 0);
    assert(numberOfPairs({1, 2, 3}, {3, 2, 1}, 0) == 3);
    assert(numberOfPairs({1}, {1}, 0) == 0);
    assert(numberOfPairs({1, 1, 1}, {1, 1, 1}, 0) == 3);

    std::cout << "number_of_pairs: all tests passed\n";
    return 0;
}
