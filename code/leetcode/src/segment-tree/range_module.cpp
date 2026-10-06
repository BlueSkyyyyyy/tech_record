// 715. Range 模块
// 见 range_module.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

struct SegTree {
    long long lo, hi;
    std::vector<int> lc, rc, lz;
    std::vector<long long> cnt;
    SegTree(long long lo_, long long hi_)
        : lo(lo_), hi(hi_), lc(2, 0), rc(2, 0), lz(2, -1), cnt(2, 0) {}

    int newNode() {
        lc.push_back(0);
        rc.push_back(0);
        lz.push_back(-1);
        cnt.push_back(0);
        return static_cast<int>(cnt.size()) - 1;
    }
    void applyNode(int o, long long l, long long r, int v) {
        cnt[o] = static_cast<long long>(v) * (r - l + 1);
        lz[o] = v;
    }
    void push(int o, long long l, long long r) {
        if (l >= r) return;
        if (!lc[o]) lc[o] = newNode();
        if (!rc[o]) rc[o] = newNode();
        if (lz[o] != -1) {
            long long m = (l + r) / 2;
            applyNode(lc[o], l, m, lz[o]);
            applyNode(rc[o], m + 1, r, lz[o]);
            lz[o] = -1;
        }
    }
    void update(int o, long long l, long long r, long long ql, long long qr, int v) {
        if (ql <= l && r <= qr) {
            applyNode(o, l, r, v);
            return;
        }
        push(o, l, r);
        long long m = (l + r) / 2;
        if (ql <= m) update(lc[o], l, m, ql, qr, v);
        if (qr > m) update(rc[o], m + 1, r, ql, qr, v);
        cnt[o] = cnt[lc[o]] + cnt[rc[o]];
    }
    long long query(int o, long long l, long long r, long long ql, long long qr) {
        if (!o) return 0;
        if (ql <= l && r <= qr) return cnt[o];
        push(o, l, r);
        long long m = (l + r) / 2, res = 0;
        if (ql <= m) res += query(lc[o], l, m, ql, qr);
        if (qr > m) res += query(rc[o], m + 1, r, ql, qr);
        return res;
    }
    void assign(long long ql, long long qr, int v) { update(1, lo, hi, ql, qr, v); }
    bool full(long long ql, long long qr) { return query(1, lo, hi, ql, qr) == qr - ql + 1; }
};

class RangeModule {
    SegTree tree;

public:
    RangeModule() : tree(1, 1000000000LL) {}
    void addRange(int left, int right) { tree.assign(left, right - 1, 1); }
    void removeRange(int left, int right) { tree.assign(left, right - 1, 0); }
    bool queryRange(int left, int right) { return tree.full(left, right - 1); }
};

int main() {
    RangeModule rm;
    rm.addRange(10, 20);
    rm.removeRange(14, 16);
    assert(rm.queryRange(10, 14) == true);
    assert(rm.queryRange(14, 16) == false);
    assert(rm.queryRange(16, 17) == true);
    assert(rm.queryRange(17, 20) == true);
    assert(rm.queryRange(10, 20) == false);
    rm.addRange(14, 16);
    assert(rm.queryRange(10, 20) == true);
    rm.removeRange(10, 20);
    assert(rm.queryRange(10, 20) == false);
    std::cout << "range_module: all tests passed\n";
    return 0;
}
