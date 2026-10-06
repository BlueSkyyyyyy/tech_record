// 587. 安装栅栏
// 见 erect_the_fence.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <set>
#include <utility>
#include <vector>

long long cross(const std::pair<int, int>& o, const std::pair<int, int>& a,
                const std::pair<int, int>& b) {
    return 1LL * (a.first - o.first) * (b.second - o.second) -
           1LL * (a.second - o.second) * (b.first - o.first);
}

std::vector<std::vector<int>> outerTrees(std::vector<std::vector<int>> points) {
    std::vector<std::pair<int, int>> pts;
    for (const auto& p : points) {
        pts.push_back({p[0], p[1]});
    }
    std::sort(pts.begin(), pts.end());
    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());

    std::vector<std::vector<int>> result;
    if (pts.size() <= 2) {
        for (const auto& p : pts) {
            result.push_back({p.first, p.second});
        }
        return result;
    }

    std::vector<std::pair<int, int>> lower;
    for (const auto& p : pts) {
        while (lower.size() >= 2 &&
               cross(lower[lower.size() - 2], lower.back(), p) < 0) {
            lower.pop_back();
        }
        lower.push_back(p);
    }

    std::vector<std::pair<int, int>> upper;
    for (auto it = pts.rbegin(); it != pts.rend(); ++it) {
        while (upper.size() >= 2 &&
               cross(upper[upper.size() - 2], upper.back(), *it) < 0) {
            upper.pop_back();
        }
        upper.push_back(*it);
    }

    std::set<std::pair<int, int>> hull(lower.begin(), lower.end());
    hull.insert(upper.begin(), upper.end());
    for (const auto& p : hull) {
        result.push_back({p.first, p.second});
    }
    return result;
}

int main() {
    std::vector<std::vector<int>> a{{1, 1}, {2, 2}, {2, 0}, {2, 4}, {3, 3}, {4, 2}};
    std::vector<std::vector<int>> aw{{1, 1}, {2, 0}, {2, 4}, {3, 3}, {4, 2}};
    assert(outerTrees(a) == aw);

    std::vector<std::vector<int>> b{{1, 2}, {2, 2}, {4, 2}};
    std::vector<std::vector<int>> bw{{1, 2}, {2, 2}, {4, 2}};
    assert(outerTrees(b) == bw);

    std::vector<std::vector<int>> c{{0, 0}, {0, 1}, {0, 2}, {1, 1}};
    std::vector<std::vector<int>> cw{{0, 0}, {0, 1}, {0, 2}, {1, 1}};
    assert(outerTrees(c) == cw);

    std::vector<std::vector<int>> d{{0, 0}, {1, 0}, {0, 1}};
    std::vector<std::vector<int>> dw{{0, 0}, {0, 1}, {1, 0}};
    assert(outerTrees(d) == dw);

    std::cout << "erect_the_fence: all tests passed\n";
    return 0;
}
