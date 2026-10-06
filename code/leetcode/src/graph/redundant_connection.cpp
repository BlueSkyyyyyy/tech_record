// 684. 冗余连接
// 见 redundant_connection.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

int findRoot(std::vector<int> &parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

std::vector<int> findRedundantConnection(std::vector<std::vector<int>> &edges) {
    int n = edges.size();
    std::vector<int> parent(n + 1);
    std::iota(parent.begin(), parent.end(), 0);

    for (auto &e : edges) {
        int u = e[0], v = e[1];
        int ru = findRoot(parent, u);
        int rv = findRoot(parent, v);
        if (ru == rv) return {u, v};
        parent[ru] = rv;
    }
    return {};
}

int main() {
    std::vector<std::vector<int>> e1 = {{1, 2}, {1, 3}, {2, 3}};
    std::vector<int> want1 = {2, 3};
    assert(findRedundantConnection(e1) == want1);

    std::vector<std::vector<int>> e2 = {{1, 2}, {2, 3}, {3, 4}, {1, 4}, {1, 5}};
    std::vector<int> want2 = {1, 4};
    assert(findRedundantConnection(e2) == want2);

    std::vector<std::vector<int>> e3 = {{1, 2}, {2, 3}, {1, 3}};
    std::vector<int> want3 = {1, 3};
    assert(findRedundantConnection(e3) == want3);

    std::cout << "redundant_connection: all tests passed\n";
    return 0;
}
