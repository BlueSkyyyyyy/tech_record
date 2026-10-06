// 765. 情侣牵手（Couples Holding Hands）
// 见 couples_holding_hands.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <unordered_set>
#include <utility>
#include <vector>

struct DSU {
    std::vector<int> parent, rank_;
    explicit DSU(int n) : parent(n), rank_(n, 0) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }
    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }
    bool unite(int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra == rb) return false;
        if (rank_[ra] < rank_[rb]) std::swap(ra, rb);
        parent[rb] = ra;
        if (rank_[ra] == rank_[rb]) ++rank_[ra];
        return true;
    }
};

int minSwapsCouples(const std::vector<int> &row) {
    int n = row.size() / 2;
    DSU dsu(n);
    for (int i = 0; i < (int)row.size(); i += 2) {
        dsu.unite(row[i] / 2, row[i + 1] / 2);
    }
    std::unordered_set<int> roots;
    for (int i = 0; i < n; ++i) roots.insert(dsu.find(i));
    return n - (int)roots.size();
}

int main() {
    std::vector<int> r1 = {0, 2, 1, 3};
    assert(minSwapsCouples(r1) == 1);

    std::vector<int> r2 = {3, 2, 0, 1};
    assert(minSwapsCouples(r2) == 0);

    std::vector<int> r3 = {0, 2, 1, 3, 4, 6, 5, 7};
    assert(minSwapsCouples(r3) == 2);

    std::vector<int> r4 = {2, 0, 1, 3};
    assert(minSwapsCouples(r4) == 1);

    std::vector<int> r5 = {0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11};
    assert(minSwapsCouples(r5) == 3);

    std::cout << "couples_holding_hands: all tests passed\n";
    return 0;
}
