// 1037. 有效的回旋镖
// 见 valid_boomerang.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

bool isBoomerang(std::vector<std::vector<int>>& points) {
    const auto& p1 = points[0];
    const auto& p2 = points[1];
    const auto& p3 = points[2];
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) -
               (p2[1] - p1[1]) * (p3[0] - p1[0]) !=
           0;
}

int main() {
    std::vector<std::vector<int>> a{{1, 1}, {2, 3}, {3, 2}};
    std::vector<std::vector<int>> b{{1, 1}, {2, 2}, {3, 3}};
    std::vector<std::vector<int>> c{{0, 0}, {0, 1}, {1, 0}};
    std::vector<std::vector<int>> d{{0, 0}, {1, 0}, {2, 0}};

    assert(isBoomerang(a) == true);
    assert(isBoomerang(b) == false);
    assert(isBoomerang(c) == true);
    assert(isBoomerang(d) == false);

    std::cout << "valid_boomerang: all tests passed\n";
    return 0;
}
