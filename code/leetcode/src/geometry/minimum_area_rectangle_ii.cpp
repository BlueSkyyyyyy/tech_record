// 963. 最小面积矩形 II
// 见 minimum_area_rectangle_ii.py 的题目与思路说明。
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

double minAreaFreeRect(std::vector<std::vector<int>>& points) {
    int n = static_cast<int>(points.size());
    std::map<std::tuple<int, int, long long>,
             std::vector<std::pair<int, int>>>
        groups;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            int mx = points[i][0] + points[j][0];
            int my = points[i][1] + points[j][1];
            long long dx = points[i][0] - points[j][0];
            long long dy = points[i][1] - points[j][1];
            long long dist2 = dx * dx + dy * dy;
            groups[{mx, my, dist2}].push_back({i, j});
        }
    }

    double best = -1.0;
    for (auto& [key, pairs] : groups) {
        int k = static_cast<int>(pairs.size());
        for (int a = 0; a < k; ++a) {
            int i = pairs[a].first, j = pairs[a].second;
            long long d1x = points[j][0] - points[i][0];
            long long d1y = points[j][1] - points[i][1];
            for (int b = a + 1; b < k; ++b) {
                int p = pairs[b].first, q = pairs[b].second;
                long long d2x = points[q][0] - points[p][0];
                long long d2y = points[q][1] - points[p][1];
                double area = std::abs(d1x * d2y - d1y * d2x) / 2.0;
                if (best < 0.0 || area < best) {
                    best = area;
                }
            }
        }
    }
    return best < 0.0 ? 0.0 : best;
}

int main() {
    std::vector<std::vector<int>> a{{0, 1}, {2, 1}, {1, 1}, {1, 0}, {2, 0}};
    std::vector<std::vector<int>> b{{1, 2}, {2, 1}, {1, 0}, {0, 1}};
    std::vector<std::vector<int>> c{{0, 0}, {1, 1}, {2, 2}};
    std::vector<std::vector<int>> d{{0, 0}, {1, 1}, {1, 0}, {0, 1}};

    assert(minAreaFreeRect(a) == 1.0);
    assert(minAreaFreeRect(b) == 2.0);
    assert(minAreaFreeRect(c) == 0.0);
    assert(minAreaFreeRect(d) == 1.0);

    std::cout << "minimum_area_rectangle_ii: all tests passed\n";
    return 0;
}
