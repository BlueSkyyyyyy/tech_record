// 685. 冗余连接 II（Redundant Connection II）
// 见 redundant_connection_ii.py 的题目与思路说明。
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

std::vector<int> findRedundantDirectedConnection(
    const std::vector<std::vector<int>> &edges) {
    int n = edges.size();
    DSU dsu(n + 1);
    std::vector<int> parent(n + 1, 0);
    int conflict = -1, cycle = -1;

    for (int i = 0; i < n; ++i) {
        int u = edges[i][0], v = edges[i][1];
        if (parent[v] != 0) {
            conflict = i;
        } else {
            parent[v] = u;
            if (!dsu.unite(u, v)) cycle = i;
        }
    }

    if (conflict == -1) return edges[cycle];
    if (cycle == -1) return edges[conflict];
    return {parent[edges[conflict][1]], edges[conflict][1]};
}

int main() {
    std::vector<std::vector<int>> e1 = {{1, 2}, {1, 3}, {2, 3}};
    std::vector<int> w1 = {2, 3};
    assert(findRedundantDirectedConnection(e1) == w1);

    std::vector<std::vector<int>> e2 = {{1, 2}, {2, 3}, {3, 4}, {4, 1}, {1, 5}};
    std::vector<int> w2 = {4, 1};
    assert(findRedundantDirectedConnection(e2) == w2);

    std::vector<std::vector<int>> e3 = {{2, 1}, {3, 1}, {4, 2}, {1, 4}};
    std::vector<int> w3 = {2, 1};
    assert(findRedundantDirectedConnection(e3) == w3);

    std::cout << "redundant_connection_ii: all tests passed\n";
    return 0;
}
