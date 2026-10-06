// 223. 矩形面积
// 见 rectangle_area.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>

int computeArea(int ax1, int ay1, int ax2, int ay2,
                int bx1, int by1, int bx2, int by2) {
    int areaA = (ax2 - ax1) * (ay2 - ay1);
    int areaB = (bx2 - bx1) * (by2 - by1);
    int width = std::max(0, std::min(ax2, bx2) - std::max(ax1, bx1));
    int height = std::max(0, std::min(ay2, by2) - std::max(ay1, by1));
    return areaA + areaB - width * height;
}

int main() {
    assert(computeArea(-3, 0, 3, 4, 0, -1, 9, 2) == 45);
    assert(computeArea(0, 0, 0, 0, 0, 0, 0, 0) == 0);
    assert(computeArea(0, 0, 2, 2, 1, 1, 3, 3) == 7);
    assert(computeArea(0, 0, 1, 1, 2, 2, 3, 3) == 2);

    std::cout << "rectangle_area: all tests passed\n";
    return 0;
}
