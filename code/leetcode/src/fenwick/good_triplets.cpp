// 2179. 统计数组中好三元组数目
// 见 good_triplets.py 的题目与思路说明。
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

long long goodTriplets(std::vector<int> nums1, std::vector<int> nums2) {
    int n = static_cast<int>(nums1.size());
    std::vector<int> pos(n, 0);
    for (int i = 0; i < n; ++i) pos[nums1[i]] = i;

    Fenwick bit(n);
    long long ans = 0;
    int leftCount = 0;
    for (int v : nums2) {
        int x = pos[v];
        long long leftLess = bit.prefix(x);
        long long leftGreater = leftCount - bit.prefix(x + 1);
        long long rightGreater = static_cast<long long>(n - 1 - x) - leftGreater;
        ans += leftLess * rightGreater;
        bit.add(x + 1, 1);
        ++leftCount;
    }
    return ans;
}

int main() {
    assert(goodTriplets({2, 0, 1, 3}, {0, 1, 2, 3}) == 1);
    assert(goodTriplets({4, 0, 1, 3, 2}, {4, 1, 0, 2, 3}) == 4);
    assert(goodTriplets({0, 1, 2}, {0, 1, 2}) == 1);
    assert(goodTriplets({0, 1, 2}, {2, 1, 0}) == 0);
    assert(goodTriplets({1, 0}, {0, 1}) == 0);

    std::cout << "good_triplets: all tests passed\n";
    return 0;
}
