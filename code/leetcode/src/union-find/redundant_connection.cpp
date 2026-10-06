// 684. 冗余连接（Redundant Connection）
// 见 redundant_connection.py 的题目与思路说明。
#include <cassert>
#include <iostream>
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

std::vector<int> findRedundantConnection(const std::vector<std::vector<int>> &edges) {
    DSU dsu(edges.size() + 1);
    for (const auto &e : edges) {
        if (!dsu.unite(e[0], e[1])) {
            return {e[0], e[1]};
        }
    }
    return {};
}

int main() {
    std::vector<std::vector<int>> e1 = {{1, 2}, {1, 3}, {2, 3}};
    std::vector<int> w1 = {2, 3};
    assert(findRedundantConnection(e1) == w1);

    std::vector<std::vector<int>> e2 = {{1, 2}, {2, 3}, {3, 4}, {1, 4}, {1, 5}};
    std::vector<int> w2 = {1, 4};
    assert(findRedundantConnection(e2) == w2);

    std::vector<std::vector<int>> e3 = {{1, 2}, {2, 3}, {1, 3}};
    std::vector<int> w3 = {1, 3};
    assert(findRedundantConnection(e3) == w3);

    std::cout << "redundant_connection: all tests passed\n";
    return 0;
}
