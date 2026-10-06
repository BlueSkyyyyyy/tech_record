// 1851. 包含每个查询的最小区间
// 见 minimum_interval_to_include_each_query.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

std::vector<int> minInterval(std::vector<std::vector<int>>& intervals,
                             std::vector<int>& queries) {
    std::sort(intervals.begin(), intervals.end());
    int q = static_cast<int>(queries.size());
    std::vector<int> order(q);
    for (int i = 0; i < q; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return queries[a] < queries[b]; });

    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>,
                        std::greater<std::pair<int, int>>>
        heap;
    std::vector<int> res(q, -1);
    int n = static_cast<int>(intervals.size());
    int j = 0;
    for (int idx : order) {
        int x = queries[idx];
        while (j < n && intervals[j][0] <= x) {
            heap.push({intervals[j][1] - intervals[j][0] + 1, intervals[j][1]});
            ++j;
        }
        while (!heap.empty() && heap.top().second < x) {
            heap.pop();
        }
        if (!heap.empty()) {
            res[idx] = heap.top().first;
        }
    }
    return res;
}

int main() {
    std::vector<std::vector<int>> a{{1, 4}, {2, 4}, {3, 6}, {4, 4}};
    std::vector<int> qa{2, 3, 4, 5};
    std::vector<int> wa{3, 3, 1, 4};
    assert(minInterval(a, qa) == wa);

    std::vector<std::vector<int>> b{{2, 3}, {2, 5}, {1, 8}, {20, 25}};
    std::vector<int> qb{2, 19, 5, 22};
    std::vector<int> wb{2, -1, 4, 6};
    assert(minInterval(b, qb) == wb);

    std::vector<std::vector<int>> e{};
    std::vector<int> qe{1, 2};
    std::vector<int> we{-1, -1};
    assert(minInterval(e, qe) == we);

    std::cout << "minimum_interval_to_include_each_query: all tests passed\n";
    return 0;
}
