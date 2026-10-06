// 218. 天际线问题
// 见 skyline.py 的题目与思路说明。
#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

std::vector<std::vector<int>> getSkyline(std::vector<std::vector<int>>& buildings) {
    std::vector<std::array<int, 3>> events;
    for (const auto& b : buildings) {
        events.push_back({b[0], -b[2], b[1]});  // 进入
        events.push_back({b[1], b[2], b[1]});   // 离开
    }
    std::sort(events.begin(), events.end());

    std::priority_queue<std::pair<int, int>> heap;  // (高度, 右端点)
    heap.push({0, INT_MAX});                         // 地面常驻
    std::vector<std::vector<int>> res;
    for (const auto& e : events) {
        int x = e[0], h = e[1], r = e[2];
        if (h < 0) {
            heap.push({-h, r});
        }
        while (heap.top().second <= x) {
            heap.pop();
        }
        int cur = heap.top().first;
        if (res.empty() || res.back()[1] != cur) {
            res.push_back({x, cur});
        }
    }
    return res;
}

int main() {
    std::vector<std::vector<int>> a{{2, 9, 10}, {3, 7, 15}, {5, 12, 12},
                                    {15, 20, 10}, {19, 24, 8}};
    std::vector<std::vector<int>> wa{{2, 10}, {3, 15}, {7, 12}, {12, 0},
                                     {15, 10}, {20, 8}, {24, 0}};
    assert(getSkyline(a) == wa);

    std::vector<std::vector<int>> b{{0, 2, 3}, {2, 5, 3}};
    std::vector<std::vector<int>> wb{{0, 3}, {5, 0}};
    assert(getSkyline(b) == wb);

    std::vector<std::vector<int>> c{{1, 2, 1}, {1, 2, 2}, {1, 2, 3}};
    std::vector<std::vector<int>> wc{{1, 3}, {2, 0}};
    assert(getSkyline(c) == wc);

    std::cout << "skyline: all tests passed\n";
    return 0;
}
