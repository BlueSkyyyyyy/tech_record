// 547. 省份数量
// 见 find_provinces.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

int findRoot(std::vector<int> &parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

int findCircleNum(std::vector<std::vector<int>> &isConnected) {
    int n = isConnected.size();
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    int count = n;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (isConnected[i][j] == 1) {
                int ri = findRoot(parent, i);
                int rj = findRoot(parent, j);
                if (ri != rj) {
                    parent[ri] = rj;
                    --count;
                }
            }
        }
    }
    return count;
}

int main() {
    std::vector<std::vector<int>> c1 = {{1, 1, 0}, {1, 1, 0}, {0, 0, 1}};
    assert(findCircleNum(c1) == 2);

    std::vector<std::vector<int>> c2 = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    assert(findCircleNum(c2) == 3);

    std::vector<std::vector<int>> c3 = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    assert(findCircleNum(c3) == 1);

    std::vector<std::vector<int>> c4 = {{1}};
    assert(findCircleNum(c4) == 1);

    std::cout << "find_provinces: all tests passed\n";
    return 0;
}
