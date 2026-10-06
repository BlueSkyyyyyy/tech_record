// 947. 移除最多的同行或同列石头（Most Stones Removed with Same Row or Column）
// 见 most_stones_removed_with_same_row_or_column.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_map>
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

int removeStones(const std::vector<std::vector<int>> &stones) {
    std::vector<int> xs, ys;
    for (const auto &s : stones) {
        xs.push_back(s[0]);
        ys.push_back(s[1]);
    }
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    std::sort(ys.begin(), ys.end());
    ys.erase(std::unique(ys.begin(), ys.end()), ys.end());

    std::unordered_map<int, int> x_id, y_id;
    for (int i = 0; i < (int)xs.size(); ++i) x_id[xs[i]] = i;
    for (int i = 0; i < (int)ys.size(); ++i) y_id[ys[i]] = (int)xs.size() + i;

    DSU dsu(xs.size() + ys.size());
    for (const auto &s : stones) {
        dsu.unite(x_id[s[0]], y_id[s[1]]);
    }

    std::unordered_set<int> roots;
    for (const auto &s : stones) roots.insert(dsu.find(x_id[s[0]]));
    return (int)stones.size() - (int)roots.size();
}

int main() {
    std::vector<std::vector<int>> s1 = {{0, 0}, {0, 1}, {1, 0}, {1, 2}, {2, 1}, {2, 2}};
    assert(removeStones(s1) == 5);

    std::vector<std::vector<int>> s2 = {{0, 0}, {0, 2}, {1, 1}, {2, 0}, {2, 2}};
    assert(removeStones(s2) == 3);

    std::vector<std::vector<int>> s3 = {{0, 0}};
    assert(removeStones(s3) == 0);

    std::cout << "most_stones_removed_with_same_row_or_column: all tests passed\n";
    return 0;
}
