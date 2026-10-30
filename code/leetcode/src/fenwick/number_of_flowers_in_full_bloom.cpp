// 2251. 花期内花的数目（差分 + 树状数组单点查询）
// 见 number_of_flowers_in_full_bloom.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

struct Fenwick {
    int n;
    std::vector<int> tree;
    explicit Fenwick(int n) : n(n), tree(n + 1, 0) {}
    void add(int i, int delta) {
        for (; i <= n; i += i & -i) tree[i] += delta;
    }
    int prefix(int i) const {
        int s = 0;
        for (; i > 0; i -= i & -i) s += tree[i];
        return s;
    }
};

std::vector<int> fullBloomFlowers(const std::vector<std::vector<int>> &flowers,
                                  const std::vector<int> &people) {
    std::vector<int> coords = people;
    for (const auto &f : flowers) {
        coords.push_back(f[0]);
        coords.push_back(f[1] + 1);
    }
    std::sort(coords.begin(), coords.end());
    coords.erase(std::unique(coords.begin(), coords.end()), coords.end());

    auto rank = [&coords](int x) {
        return static_cast<int>(std::lower_bound(coords.begin(), coords.end(), x) - coords.begin()) + 1;
    };

    Fenwick bit(static_cast<int>(coords.size()));
    for (const auto &f : flowers) {
        bit.add(rank(f[0]), 1);
        bit.add(rank(f[1] + 1), -1);
    }

    std::vector<int> res;
    res.reserve(people.size());
    for (int t : people) res.push_back(bit.prefix(rank(t)));
    return res;
}

int main() {
    std::vector<int> expected = {1, 2, 2, 2};
    assert(fullBloomFlowers({{1, 6}, {3, 7}, {9, 12}, {4, 13}}, {2, 3, 7, 11}) == expected);

    std::vector<int> expected2 = {2, 2, 1};
    assert(fullBloomFlowers({{1, 10}, {3, 3}}, {3, 3, 2}) == expected2);

    std::vector<int> expected3 = {1, 0};
    assert(fullBloomFlowers({{1, 1}}, {1, 2}) == expected3);

    std::vector<int> expected4 = {0, 0};
    assert(fullBloomFlowers({{1, 5}}, {0, 6}) == expected4);

    std::cout << "number_of_flowers_in_full_bloom: all tests passed\n";
    return 0;
}
