// 547. 省份数量（Number of Provinces）
// 见 number_of_provinces.py 的题目与思路说明。
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

int findCircleNum(const std::vector<std::vector<int>> &isConnected) {
    int n = isConnected.size();
    DSU dsu(n);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (isConnected[i][j] == 1) {
                dsu.unite(i, j);
            }
        }
    }
    std::unordered_set<int> roots;
    for (int i = 0; i < n; ++i) roots.insert(dsu.find(i));
    return roots.size();
}

int main() {
    std::vector<std::vector<int>> c1 = {{1, 1, 0}, {1, 1, 0}, {0, 0, 1}};
    assert(findCircleNum(c1) == 2);

    std::vector<std::vector<int>> c2 = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    assert(findCircleNum(c2) == 3);

    std::vector<std::vector<int>> c3 = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    assert(findCircleNum(c3) == 1);

    std::vector<std::vector<int>> c4 = {{1}};
    assert(findCircleNum(c4) == 1);

    std::cout << "number_of_provinces: all tests passed\n";
    return 0;
}
