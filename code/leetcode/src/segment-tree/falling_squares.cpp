// 699. 掉落的方块
// 见 falling_squares.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <vector>

struct SegTree {
    int n;
    std::vector<int> mx, lz;
    explicit SegTree(int n_) : n(n_), mx(4 * n_, 0), lz(4 * n_, -1) {}

    void applyNode(int o, int v) {
        mx[o] = v;
        lz[o] = v;
    }
    void push(int o) {
        if (lz[o] != -1) {
            applyNode(2 * o, lz[o]);
            applyNode(2 * o + 1, lz[o]);
            lz[o] = -1;
        }
    }
    void update(int o, int l, int r, int ql, int qr, int v) {
        if (ql <= l && r <= qr) {
            applyNode(o, v);
            return;
        }
        push(o);
        int m = (l + r) / 2;
        if (ql <= m) update(2 * o, l, m, ql, qr, v);
        if (qr > m) update(2 * o + 1, m + 1, r, ql, qr, v);
        mx[o] = std::max(mx[2 * o], mx[2 * o + 1]);
    }
    int query(int o, int l, int r, int ql, int qr) {
        if (ql <= l && r <= qr) return mx[o];
        push(o);
        int m = (l + r) / 2, res = 0;
        if (ql <= m) res = std::max(res, query(2 * o, l, m, ql, qr));
        if (qr > m) res = std::max(res, query(2 * o + 1, m + 1, r, ql, qr));
        return res;
    }
    void assign(int ql, int qr, int v) { update(1, 0, n - 1, ql, qr, v); }
    int query(int ql, int qr) { return query(1, 0, n - 1, ql, qr); }
};

std::vector<int> fallingSquares(const std::vector<std::pair<int, int>>& positions) {
    std::vector<int> xs;
    for (auto [left, side] : positions) {
        xs.push_back(left);
        xs.push_back(left + side);
    }
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    std::map<int, int> idx;
    for (int i = 0; i < static_cast<int>(xs.size()); ++i) idx[xs[i]] = i;

    SegTree st(static_cast<int>(xs.size()) - 1);
    std::vector<int> res;
    int cur = 0;
    for (auto [left, side] : positions) {
        int li = idx[left];
        int ri = idx[left + side] - 1;
        int base = st.query(li, ri);
        int h = base + side;
        st.assign(li, ri, h);
        cur = std::max(cur, h);
        res.push_back(cur);
    }
    return res;
}

int main() {
    assert(fallingSquares({{1, 2}, {2, 3}, {6, 1}}) == (std::vector<int>{2, 5, 5}));
    assert(fallingSquares({{100, 100}, {200, 100}}) == (std::vector<int>{100, 100}));
    assert(fallingSquares({{1, 1}, {1, 1}, {1, 1}}) == (std::vector<int>{1, 2, 3}));
    std::cout << "falling_squares: all tests passed\n";
    return 0;
}
