// 731. 我的日程安排表 II
// 见 my_calendar_ii.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

struct SegTree {
    long long lo, hi;
    std::vector<int> lc, rc, mx, lz;
    SegTree(long long lo_, long long hi_)
        : lo(lo_), hi(hi_), lc(2, 0), rc(2, 0), mx(2, 0), lz(2, 0) {}

    int newNode() {
        lc.push_back(0);
        rc.push_back(0);
        mx.push_back(0);
        lz.push_back(0);
        return static_cast<int>(mx.size()) - 1;
    }
    void applyNode(int o, int v) {
        mx[o] += v;
        lz[o] += v;
    }
    void push(int o, long long l, long long r) {
        if (l >= r) return;
        if (!lc[o]) lc[o] = newNode();
        if (!rc[o]) rc[o] = newNode();
        if (lz[o]) {
            applyNode(lc[o], lz[o]);
            applyNode(rc[o], lz[o]);
            lz[o] = 0;
        }
    }
    void update(int o, long long l, long long r, long long ql, long long qr, int v) {
        if (ql <= l && r <= qr) {
            applyNode(o, v);
            return;
        }
        push(o, l, r);
        long long m = (l + r) / 2;
        if (ql <= m) update(lc[o], l, m, ql, qr, v);
        if (qr > m) update(rc[o], m + 1, r, ql, qr, v);
        mx[o] = std::max(mx[lc[o]], mx[rc[o]]);
    }
    int query(int o, long long l, long long r, long long ql, long long qr) {
        if (!o) return 0;
        if (ql <= l && r <= qr) return mx[o];
        push(o, l, r);
        long long m = (l + r) / 2;
        int res = 0;
        if (ql <= m) res = std::max(res, query(lc[o], l, m, ql, qr));
        if (qr > m) res = std::max(res, query(rc[o], m + 1, r, ql, qr));
        return res;
    }
    void add(long long ql, long long qr, int v) {
        if (ql <= qr) update(1, lo, hi, ql, qr, v);
    }
    int query(long long ql, long long qr) {
        if (ql > qr) return 0;
        return query(1, lo, hi, ql, qr);
    }
};

class MyCalendarTwo {
    SegTree tree;

public:
    MyCalendarTwo() : tree(0, 1000000000LL) {}
    bool book(int start, int end) {
        if (tree.query(start, end - 1) >= 2) return false;
        tree.add(start, end - 1, 1);
        return true;
    }
};

int main() {
    MyCalendarTwo cal;
    assert(cal.book(10, 20) == true);
    assert(cal.book(50, 60) == true);
    assert(cal.book(10, 40) == true);
    assert(cal.book(5, 15) == false);
    assert(cal.book(5, 10) == true);
    assert(cal.book(25, 55) == true);
    std::cout << "my_calendar_ii: all tests passed\n";
    return 0;
}
