// 939. 最小面积矩形
// 见 minimum_area_rectangle.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

int minAreaRect(std::vector<std::vector<int>>& points) {
    std::map<int, std::vector<int>> byX;
    for (const auto& p : points) {
        byX[p[0]].push_back(p[1]);
    }

    std::map<std::pair<int, int>, int> last;
    int best = INT_MAX;
    for (auto& [x, ys] : byX) {
        std::sort(ys.begin(), ys.end());
        int m = static_cast<int>(ys.size());
        for (int i = 0; i < m; ++i) {
            for (int j = i + 1; j < m; ++j) {
                std::pair<int, int> key{ys[i], ys[j]};
                auto it = last.find(key);
                if (it != last.end()) {
                    int area = (x - it->second) * (ys[j] - ys[i]);
                    best = std::min(best, area);
                }
                last[key] = x;
            }
        }
    }
    return best == INT_MAX ? 0 : best;
}

int main() {
    std::vector<std::vector<int>> a{{1, 1}, {1, 3}, {3, 1}, {3, 3}, {2, 2}};
    std::vector<std::vector<int>> b{{1, 1}, {1, 3}, {3, 1}, {3, 3}, {4, 1}, {4, 3}};
    std::vector<std::vector<int>> c{{1, 1}, {2, 2}, {3, 3}};

    assert(minAreaRect(a) == 4);
    assert(minAreaRect(b) == 2);
    assert(minAreaRect(c) == 0);

    std::cout << "minimum_area_rectangle: all tests passed\n";
    return 0;
}
