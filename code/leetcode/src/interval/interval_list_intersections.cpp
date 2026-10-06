// 986. 区间列表的交集
// 见 interval_list_intersections.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

std::vector<std::vector<int>> intervalIntersection(
    std::vector<std::vector<int>>& first, std::vector<std::vector<int>>& second) {
    std::vector<std::vector<int>> res;
    int i = 0, j = 0;
    int m = static_cast<int>(first.size());
    int n = static_cast<int>(second.size());
    while (i < m && j < n) {
        int lo = std::max(first[i][0], second[j][0]);
        int hi = std::min(first[i][1], second[j][1]);
        if (lo <= hi) {
            res.push_back({lo, hi});
        }
        if (first[i][1] < second[j][1]) {
            ++i;
        } else {
            ++j;
        }
    }
    return res;
}

int main() {
    std::vector<std::vector<int>> first{{0, 2}, {5, 10}, {13, 23}, {24, 25}};
    std::vector<std::vector<int>> second{{1, 5}, {8, 12}, {15, 24}, {25, 26}};
    std::vector<std::vector<int>> want{{1, 2},  {5, 5},   {8, 10},
                                       {15, 23}, {24, 24}, {25, 25}};

    assert(intervalIntersection(first, second) == want);

    std::vector<std::vector<int>> c{{1, 3}, {5, 9}};
    std::vector<std::vector<int>> d{};
    assert(intervalIntersection(c, d).empty());

    std::cout << "interval_list_intersections: all tests passed\n";
    return 0;
}
