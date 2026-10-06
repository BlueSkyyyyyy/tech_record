// 57. 插入区间
// 见 insert_interval.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

std::vector<std::vector<int>> insert(std::vector<std::vector<int>>& intervals,
                                     std::vector<int>& newInterval) {
    std::vector<std::vector<int>> res;
    int n = static_cast<int>(intervals.size());
    int i = 0;
    int start = newInterval[0], end = newInterval[1];

    while (i < n && intervals[i][1] < start) {
        res.push_back(intervals[i]);
        ++i;
    }
    while (i < n && intervals[i][0] <= end) {
        start = std::min(start, intervals[i][0]);
        end = std::max(end, intervals[i][1]);
        ++i;
    }
    res.push_back({start, end});
    while (i < n) {
        res.push_back(intervals[i]);
        ++i;
    }
    return res;
}

int main() {
    std::vector<std::vector<int>> a{{1, 3}, {6, 9}};
    std::vector<int> na{2, 5};
    std::vector<std::vector<int>> wa{{1, 5}, {6, 9}};

    std::vector<std::vector<int>> b{{1, 2}, {3, 5}, {6, 7}, {8, 10}, {12, 16}};
    std::vector<int> nb{4, 8};
    std::vector<std::vector<int>> wb{{1, 2}, {3, 10}, {12, 16}};

    std::vector<std::vector<int>> e{};
    std::vector<int> ne{5, 7};
    std::vector<std::vector<int>> we{{5, 7}};

    assert(insert(a, na) == wa);
    assert(insert(b, nb) == wb);
    assert(insert(e, ne) == we);

    std::cout << "insert_interval: all tests passed\n";
    return 0;
}
