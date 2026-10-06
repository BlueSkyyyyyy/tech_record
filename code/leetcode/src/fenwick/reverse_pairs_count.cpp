// LCR 170. 交易逆序对总数（原剑指 Offer 51. 数组中的逆序对）
// 见 reverse_pairs_count.py 的题目与思路说明。
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

long long reversePairsCount(std::vector<int> record) {
    std::vector<int> vals = record;
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

    Fenwick bit(static_cast<int>(vals.size()));
    long long ans = 0;
    int processed = 0;
    for (int x : record) {
        int r = static_cast<int>(
                    std::lower_bound(vals.begin(), vals.end(), x) - vals.begin()) +
                1;
        ans += processed - bit.prefix(r);
        bit.add(r, 1);
        ++processed;
    }
    return ans;
}

int main() {
    assert(reversePairsCount({7, 5, 6, 4}) == 5);
    assert(reversePairsCount({1, 2, 3, 4}) == 0);
    assert(reversePairsCount({4, 3, 2, 1}) == 6);
    assert(reversePairsCount({1}) == 0);
    assert(reversePairsCount({}) == 0);
    assert(reversePairsCount({-1, -2, 0, -1}) == 2);
    assert(reversePairsCount({2, 2, 2}) == 0);

    std::cout << "reverse_pairs_count: all tests passed\n";
    return 0;
}
