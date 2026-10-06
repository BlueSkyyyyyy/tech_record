// 1409. 查询带键的排列
// 见 process_queries.py 的题目与思路说明。
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

std::vector<int> processQueries(std::vector<int> queries, int m) {
    int n = static_cast<int>(queries.size());
    Fenwick bit(m + n);

    std::vector<int> pos(m + 1, 0);
    for (int v = 1; v <= m; ++v) {
        pos[v] = n + v;
        bit.add(pos[v], 1);
    }

    std::vector<int> ans;
    ans.reserve(n);
    int nextPos = n;
    for (int q : queries) {
        int p = pos[q];
        ans.push_back(static_cast<int>(bit.prefix(p - 1)));
        bit.add(p, -1);
        bit.add(nextPos, 1);
        pos[q] = nextPos;
        --nextPos;
    }
    return ans;
}

int main() {
    std::vector<int> got1 = processQueries({3, 1, 2, 1}, 5);
    std::vector<int> want1 = {2, 1, 2, 1};
    assert(got1 == want1);

    std::vector<int> got2 = processQueries({4, 1, 2, 2}, 4);
    std::vector<int> want2 = {3, 1, 2, 0};
    assert(got2 == want2);

    std::vector<int> got3 = processQueries({7, 5, 5, 8, 3}, 8);
    std::vector<int> want3 = {6, 5, 0, 7, 5};
    assert(got3 == want3);

    std::vector<int> got4 = processQueries({1}, 1);
    std::vector<int> want4 = {0};
    assert(got4 == want4);

    std::cout << "process_queries: all tests passed\n";
    return 0;
}
