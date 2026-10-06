// 990. 等式方程的可满足性（Satisfiability of Equality Equations）
// 见 satisfiability_of_equality_equations.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
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

bool equationsPossible(const std::vector<std::string> &equations) {
    DSU dsu(26);
    for (const auto &eq : equations) {
        if (eq[1] == '=') {
            dsu.unite(eq[0] - 'a', eq[3] - 'a');
        }
    }
    for (const auto &eq : equations) {
        if (eq[1] == '!' && dsu.find(eq[0] - 'a') == dsu.find(eq[3] - 'a')) {
            return false;
        }
    }
    return true;
}

int main() {
    std::vector<std::string> e1 = {"a==b", "b!=a"};
    assert(equationsPossible(e1) == false);

    std::vector<std::string> e2 = {"b==a", "a==b"};
    assert(equationsPossible(e2) == true);

    std::vector<std::string> e3 = {"a==b", "b==c", "a==c"};
    assert(equationsPossible(e3) == true);

    std::vector<std::string> e4 = {"a==b", "b!=c", "c==a"};
    assert(equationsPossible(e4) == false);

    std::vector<std::string> e5 = {"c==c", "b==d", "x!=z"};
    assert(equationsPossible(e5) == true);

    std::cout << "satisfiability_of_equality_equations: all tests passed\n";
    return 0;
}
