// 2569. 更新数组后处理求和查询
// 见 handling_sum_queries.py 的题目与思路说明。
#include <array>
#include <cassert>
#include <iostream>
#include <vector>

struct SegTree {
    int n;
    std::vector<long long> s;
    std::vector<bool> lz;
    explicit SegTree(const std::vector<int>& nums) : n(nums.size()), s(4 * n, 0), lz(4 * n, false) {
        build(1, 0, n - 1, nums);
    }
    void build(int o, int l, int r, const std::vector<int>& nums) {
        if (l == r) {
            s[o] = nums[l];
            return;
        }
        int m = (l + r) / 2;
        build(2 * o, l, m, nums);
        build(2 * o + 1, m + 1, r, nums);
        s[o] = s[2 * o] + s[2 * o + 1];
    }
    void applyNode(int o, int l, int r) {
        s[o] = (r - l + 1) - s[o];
        lz[o] = !lz[o];
    }
    void push(int o, int l, int r) {
        if (lz[o]) {
            int m = (l + r) / 2;
            applyNode(2 * o, l, m);
            applyNode(2 * o + 1, m + 1, r);
            lz[o] = false;
        }
    }
    void flip(int o, int l, int r, int ql, int qr) {
        if (ql <= l && r <= qr) {
            applyNode(o, l, r);
            return;
        }
        push(o, l, r);
        int m = (l + r) / 2;
        if (ql <= m) flip(2 * o, l, m, ql, qr);
        if (qr > m) flip(2 * o + 1, m + 1, r, ql, qr);
        s[o] = s[2 * o] + s[2 * o + 1];
    }
    long long query(int o, int l, int r, int ql, int qr) {
        if (ql <= l && r <= qr) return s[o];
        push(o, l, r);
        int m = (l + r) / 2;
        long long res = 0;
        if (ql <= m) res += query(2 * o, l, m, ql, qr);
        if (qr > m) res += query(2 * o + 1, m + 1, r, ql, qr);
        return res;
    }
    void flip(int ql, int qr) { flip(1, 0, n - 1, ql, qr); }
    long long query(int ql, int qr) { return query(1, 0, n - 1, ql, qr); }
};

std::vector<long long> handleQueries(const std::vector<int>& nums1, const std::vector<int>& nums2,
                                     const std::vector<std::array<long long, 3>>& queries) {
    SegTree st(nums1);
    long long total = 0;
    for (int v : nums2) total += v;
    std::vector<long long> res;
    for (auto& q : queries) {
        if (q[0] == 1) {
            st.flip(static_cast<int>(q[1]), static_cast<int>(q[2]));
        } else if (q[0] == 2) {
            total += q[1] * st.query(0, static_cast<int>(nums1.size()) - 1);
        } else {
            res.push_back(total);
        }
    }
    return res;
}

int main() {
    assert(handleQueries({1, 0, 1}, {0, 0, 0}, {{1, 1, 1}, {2, 1, 0}, {3, 0, 0}}) ==
           (std::vector<long long>{3}));
    assert(handleQueries({1}, {5}, {{2, 0, 0}, {3, 0, 0}}) == (std::vector<long long>{5}));
    assert(handleQueries({1, 0, 1}, {0, 0, 0}, {{2, 1, 0}, {3, 0, 0}}) ==
           (std::vector<long long>{2}));
    std::cout << "handling_sum_queries: all tests passed\n";
    return 0;
}
