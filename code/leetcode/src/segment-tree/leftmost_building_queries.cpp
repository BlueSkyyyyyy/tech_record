// 2940. 找到 Alice 和 Bob 可以相遇的建筑
// 见 leftmost_building_queries.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <vector>

std::vector<int> leftmostBuildingQueries(const std::vector<int>& heights,
                                         const std::vector<std::pair<int, int>>& queries) {
    int n = heights.size();
    int size = 1;
    while (size < n) size *= 2;
    std::vector<int> tree(2 * size, -1);
    for (int i = 0; i < n; ++i) tree[size + i] = heights[i];
    for (int i = size - 1; i > 0; --i) tree[i] = std::max(tree[2 * i], tree[2 * i + 1]);

    auto rangeMax = [&](int l, int r) {
        int res = -1;
        for (l += size, r += size + 1; l < r; l /= 2, r /= 2) {
            if (l & 1) res = std::max(res, tree[l++]);
            if (r & 1) res = std::max(res, tree[--r]);
        }
        return res;
    };
    auto firstGreater = [&](int l, int t) -> int {
        if (l >= n || rangeMax(l, n - 1) <= t) return -1;
        std::function<int(int, int, int)> rec = [&](int o, int nl, int nr) -> int {
            if (nr < l || tree[o] <= t) return -1;
            if (nl == nr) return nl;
            int mid = (nl + nr) / 2;
            int left = rec(2 * o, nl, mid);
            if (left != -1) return left;
            return rec(2 * o + 1, mid + 1, nr);
        };
        return rec(1, 0, size - 1);
    };

    std::vector<int> res;
    for (auto [a, b] : queries) {
        if (a == b) {
            res.push_back(a);
            continue;
        }
        int lo = std::min(a, b), hi = std::max(a, b);
        if (heights[hi] > heights[lo]) {
            res.push_back(hi);
        } else {
            res.push_back(firstGreater(hi + 1, std::max(heights[a], heights[b])));
        }
    }
    return res;
}

int main() {
    assert(leftmostBuildingQueries({6, 4, 8, 5, 2, 7}, {{0, 1}, {0, 3}, {2, 4}, {3, 4}, {2, 2}}) ==
           (std::vector<int>{2, 5, -1, 5, 2}));
    assert(leftmostBuildingQueries({5, 3, 8, 2, 6, 1, 4, 6},
                                   {{0, 7}, {3, 5}, {5, 2}, {3, 0}, {1, 6}}) ==
           (std::vector<int>{7, 6, -1, 4, 6}));
    assert(leftmostBuildingQueries({1, 2, 1}, {{1, 0}}) == (std::vector<int>{1}));
    std::cout << "leftmost_building_queries: all tests passed\n";
    return 0;
}
