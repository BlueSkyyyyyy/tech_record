// 452. 用最少数量的箭引爆气球
// 见 minimum_number_of_arrows.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int findMinArrowShots(std::vector<std::vector<int>> points) {
    if (points.empty()) return 0;
    std::sort(points.begin(), points.end(),
              [](const std::vector<int> &a, const std::vector<int> &b) {
                  return a[1] < b[1];
              });
    int arrows = 1;
    int end = points[0][1];
    for (const auto &p : points) {
        if (p[0] > end) {
            ++arrows;
            end = p[1];
        }
    }
    return arrows;
}

int main() {
    std::vector<std::vector<int>> a = {{10, 16}, {2, 8}, {1, 6}, {7, 12}};
    assert(findMinArrowShots(a) == 2);
    std::vector<std::vector<int>> b = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    assert(findMinArrowShots(b) == 4);
    std::vector<std::vector<int>> c = {{1, 2}, {2, 3}, {3, 4}, {4, 5}};
    assert(findMinArrowShots(c) == 2);
    std::vector<std::vector<int>> d;
    assert(findMinArrowShots(d) == 0);
    std::vector<std::vector<int>> e = {{1, 2}};
    assert(findMinArrowShots(e) == 1);
    std::cout << "minimum_number_of_arrows: all tests passed\n";
    return 0;
}
