// 785. 判断二分图（Is Graph Bipartite?）
// 见 is_graph_bipartite.py 的题目与思路说明。
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

bool isBipartite(const std::vector<std::vector<int>> &graph) {
    int n = graph.size();
    DSU dsu(n);
    for (int u = 0; u < n; ++u) {
        for (int v : graph[u]) {
            if (dsu.find(u) == dsu.find(v)) return false;
            dsu.unite(graph[u][0], v);
        }
    }
    return true;
}

int main() {
    std::vector<std::vector<int>> g1 = {{1, 2, 3}, {0, 2}, {0, 1, 3}, {0, 2}};
    assert(isBipartite(g1) == false);

    std::vector<std::vector<int>> g2 = {{1, 3}, {0, 2}, {1, 3}, {0, 2}};
    assert(isBipartite(g2) == true);

    std::vector<std::vector<int>> g3 = {{}};
    assert(isBipartite(g3) == true);

    std::vector<std::vector<int>> g4 = {{1}, {0}};
    assert(isBipartite(g4) == true);

    std::vector<std::vector<int>> g5 = {{1, 2}, {0}, {0}};
    assert(isBipartite(g5) == true);

    std::cout << "is_graph_bipartite: all tests passed\n";
    return 0;
}
