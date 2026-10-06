// 836. 矩形重叠
// 见 rectangle_overlap.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

bool isRectangleOverlap(std::vector<int>& rec1, std::vector<int>& rec2) {
    bool x = std::min(rec1[2], rec2[2]) > std::max(rec1[0], rec2[0]);
    bool y = std::min(rec1[3], rec2[3]) > std::max(rec1[1], rec2[1]);
    return x && y;
}

int main() {
    std::vector<int> a0{0, 0, 2, 2}, b0{1, 1, 3, 3};
    std::vector<int> a1{0, 0, 1, 1}, b1{1, 0, 2, 1};
    std::vector<int> a2{0, 0, 1, 1}, b2{2, 2, 3, 3};
    std::vector<int> a3{0, 0, 3, 3}, b3{1, 1, 2, 2};
    std::vector<int> a4{7, 8, 13, 15}, b4{10, 8, 12, 20};

    assert(isRectangleOverlap(a0, b0) == true);
    assert(isRectangleOverlap(a1, b1) == false);
    assert(isRectangleOverlap(a2, b2) == false);
    assert(isRectangleOverlap(a3, b3) == true);
    assert(isRectangleOverlap(a4, b4) == true);

    std::cout << "rectangle_overlap: all tests passed\n";
    return 0;
}
