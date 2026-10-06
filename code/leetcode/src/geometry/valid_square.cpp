// 593. 有效的正方形
// 见 valid_square.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

bool validSquare(std::vector<int>& p1, std::vector<int>& p2,
                 std::vector<int>& p3, std::vector<int>& p4) {
    std::vector<std::vector<int>> pts{p1, p2, p3, p4};
    std::vector<long long> dists;
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            long long dx = pts[i][0] - pts[j][0];
            long long dy = pts[i][1] - pts[j][1];
            dists.push_back(dx * dx + dy * dy);
        }
    }
    std::sort(dists.begin(), dists.end());
    long long side = dists[0];
    long long diag = dists[4];
    return side > 0 && dists[0] == dists[1] && dists[1] == dists[2] &&
           dists[2] == dists[3] && diag == dists[5] && diag == 2 * side;
}

int main() {
    std::vector<int> a0{0, 0}, a1{1, 1}, a2{1, 0}, a3{0, 1};
    std::vector<int> b0{0, 0}, b1{1, 1}, b2{1, 0}, b3{0, 12};
    std::vector<int> c0{1, 0}, c1{-1, 0}, c2{0, 1}, c3{0, -1};
    std::vector<int> d0{0, 0}, d1{0, 0}, d2{0, 0}, d3{0, 0};

    assert(validSquare(a0, a1, a2, a3) == true);
    assert(validSquare(b0, b1, b2, b3) == false);
    assert(validSquare(c0, c1, c2, c3) == true);
    assert(validSquare(d0, d1, d2, d3) == false);

    std::cout << "valid_square: all tests passed\n";
    return 0;
}
