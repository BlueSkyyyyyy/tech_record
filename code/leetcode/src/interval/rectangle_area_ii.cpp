// 850. 矩形面积 II
// 见 rectangle_area_ii.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>

int rectangleArea(std::vector<std::vector<int>>& rectangles) {
    const long long MOD = 1000000007LL;
    std::vector<long long> xs;
    for (const auto& r : rectangles) {
        xs.push_back(r[0]);
        xs.push_back(r[2]);
    }
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());

    long long area = 0;
    for (size_t i = 0; i + 1 < xs.size(); ++i) {
        long long xa = xs[i], xb = xs[i + 1];
        std::vector<std::pair<long long, long long>> spans;
        for (const auto& r : rectangles) {
            if (r[0] <= xa && xb <= r[2]) {
                spans.push_back({r[1], r[3]});
            }
        }
        std::sort(spans.begin(), spans.end());

        long long covered = 0;
        bool has = false;
        long long lo = 0, hi = 0;
        for (const auto& [y1, y2] : spans) {
            if (!has) {
                lo = y1;
                hi = y2;
                has = true;
            } else if (y1 > hi) {
                covered += hi - lo;
                lo = y1;
                hi = y2;
            } else {
                hi = std::max(hi, y2);
            }
        }
        if (has) {
            covered += hi - lo;
        }
        area = (area + (xb - xa) % MOD * (covered % MOD)) % MOD;
    }
    return static_cast<int>(area);
}

int main() {
    std::vector<std::vector<int>> a{{0, 0, 2, 2}, {1, 0, 2, 3}, {1, 0, 3, 1}};
    assert(rectangleArea(a) == 6);

    std::vector<std::vector<int>> b{{0, 0, 1000000000, 1000000000}};
    assert(rectangleArea(b) == 49);

    std::vector<std::vector<int>> c{{0, 0, 1, 1}, {2, 2, 3, 3}};
    assert(rectangleArea(c) == 2);

    std::cout << "rectangle_area_ii: all tests passed\n";
    return 0;
}
