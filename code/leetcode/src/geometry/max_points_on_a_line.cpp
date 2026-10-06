// 149. 直线上最多的点数
// 见 max_points_on_a_line.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

int maxPoints(std::vector<std::vector<int>>& points) {
    int n = static_cast<int>(points.size());
    if (n <= 2) {
        return n;
    }
    int best = 0;
    for (int i = 0; i < n; ++i) {
        std::map<std::pair<int, int>, int> directions;
        for (int j = i + 1; j < n; ++j) {
            int dx = points[j][0] - points[i][0];
            int dy = points[j][1] - points[i][1];
            int g = std::gcd(std::abs(dx), std::abs(dy));
            if (g == 0) {
                g = 1;
            }
            dx /= g;
            dy /= g;
            if (dx < 0 || (dx == 0 && dy < 0)) {
                dx = -dx;
                dy = -dy;
            }
            ++directions[{dx, dy}];
        }
        for (const auto& kv : directions) {
            best = std::max(best, 1 + kv.second);
        }
    }
    return best;
}

int main() {
    std::vector<std::vector<int>> a{{1, 1}, {2, 2}, {3, 3}};
    std::vector<std::vector<int>> b{{1, 1}, {3, 2}, {5, 3}, {4, 1}, {2, 3}, {1, 4}};
    std::vector<std::vector<int>> c{{0, 0}};
    std::vector<std::vector<int>> d{{0, 0}, {1, 1}};
    std::vector<std::vector<int>> e{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}};

    assert(maxPoints(a) == 3);
    assert(maxPoints(b) == 4);
    assert(maxPoints(c) == 1);
    assert(maxPoints(d) == 2);
    assert(maxPoints(e) == 4);

    std::cout << "max_points_on_a_line: all tests passed\n";
    return 0;
}
