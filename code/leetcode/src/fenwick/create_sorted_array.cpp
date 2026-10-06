// 1649. 通过指令创建有序数组
// 见 create_sorted_array.py 的题目与思路说明。
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

int createSortedArray(std::vector<int> instructions) {
    const long long MOD = 1000000007LL;
    const int MAXV = 100000;

    Fenwick bit(MAXV);
    long long ans = 0;
    int processed = 0;
    for (int x : instructions) {
        long long less = bit.prefix(x - 1);
        long long greater = processed - bit.prefix(x);
        ans = (ans + std::min(less, greater)) % MOD;
        bit.add(x, 1);
        ++processed;
    }
    return static_cast<int>(ans);
}

int main() {
    assert(createSortedArray({1, 5, 6, 2}) == 1);
    assert(createSortedArray({1, 2, 3, 6, 5, 4}) == 3);
    assert(createSortedArray({1, 2, 3, 4}) == 0);
    assert(createSortedArray({4, 3, 2, 1}) == 0);
    assert(createSortedArray({5}) == 0);

    std::cout << "create_sorted_array: all tests passed\n";
    return 0;
}
