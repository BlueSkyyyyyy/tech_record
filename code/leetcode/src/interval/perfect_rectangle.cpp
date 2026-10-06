// 391. 完美矩形
// 见 perfect_rectangle.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>
#include <set>
#include <utility>
#include <vector>

bool isRectangleCover(std::vector<std::vector<int>>& rectangles) {
    long long area = 0;
    int minX = INT_MAX, minY = INT_MAX;
    int maxX = INT_MIN, maxY = INT_MIN;
    std::set<std::pair<int, int>> corners;

    for (const auto& r : rectangles) {
        int x1 = r[0], y1 = r[1], x2 = r[2], y2 = r[3];
        area += 1LL * (x2 - x1) * (y2 - y1);
        minX = std::min(minX, x1);
        minY = std::min(minY, y1);
        maxX = std::max(maxX, x2);
        maxY = std::max(maxY, y2);
        std::pair<int, int> pts[4] = {{x1, y1}, {x1, y2}, {x2, y1}, {x2, y2}};
        for (const auto& p : pts) {
            if (corners.count(p)) {
                corners.erase(p);
            } else {
                corners.insert(p);
            }
        }
    }

    if (area != 1LL * (maxX - minX) * (maxY - minY)) {
        return false;
    }
    std::set<std::pair<int, int>> expected{
        {minX, minY}, {minX, maxY}, {maxX, minY}, {maxX, maxY}};
    return corners == expected;
}

int main() {
    std::vector<std::vector<int>> a{
        {1, 1, 3, 3}, {3, 1, 4, 2}, {3, 2, 4, 3}, {1, 3, 4, 4}};
    assert(isRectangleCover(a) == true);

    std::vector<std::vector<int>> b{
        {1, 1, 2, 3}, {1, 3, 2, 4}, {3, 1, 4, 2}, {3, 2, 4, 4}};
    assert(isRectangleCover(b) == false);

    std::vector<std::vector<int>> c{
        {1, 1, 3, 3}, {3, 1, 4, 2}, {1, 3, 2, 4}, {2, 2, 4, 4}};
    assert(isRectangleCover(c) == false);

    std::vector<std::vector<int>> d{
        {0, 0, 1, 1}, {0, 1, 1, 2}, {0, 2, 1, 3}, {0, 3, 1, 4}};
    assert(isRectangleCover(d) == true);

    std::cout << "perfect_rectangle: all tests passed\n";
    return 0;
}
