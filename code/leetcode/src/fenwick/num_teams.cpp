// 1395. 统计作战单位数
// 见 num_teams.py 的题目与思路说明。
#include <algorithm>
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

int numTeams(std::vector<int> rating) {
    int maxv = 0;
    for (int x : rating) maxv = std::max(maxv, x);

    std::vector<int> total(maxv + 2, 0);
    for (int x : rating) ++total[x];

    std::vector<int> lessTotal(maxv + 2, 0), greaterTotal(maxv + 2, 0);
    int run = 0;
    for (int v = 1; v <= maxv; ++v) {
        lessTotal[v] = run;
        run += total[v];
    }
    run = 0;
    for (int v = maxv; v >= 1; --v) {
        greaterTotal[v] = run;
        run += total[v];
    }

    Fenwick bit(maxv);
    long long ans = 0;
    int leftCount = 0;
    for (int x : rating) {
        long long leftLess = bit.prefix(x - 1);
        long long leftGreater = leftCount - bit.prefix(x);
        long long rightLess = lessTotal[x] - leftLess;
        long long rightGreater = greaterTotal[x] - leftGreater;
        ans += leftLess * rightGreater + leftGreater * rightLess;
        bit.add(x, 1);
        ++leftCount;
    }
    return static_cast<int>(ans);
}

int main() {
    assert(numTeams({2, 5, 3, 4, 1}) == 3);
    assert(numTeams({2, 1, 3}) == 0);
    assert(numTeams({1, 2, 3, 4}) == 4);
    assert(numTeams({4, 3, 2, 1}) == 4);
    assert(numTeams({1, 3, 2, 4}) == 2);

    std::cout << "num_teams: all tests passed\n";
    return 0;
}
