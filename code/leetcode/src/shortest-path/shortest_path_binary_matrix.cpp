// 1091. 二进制矩阵中的最短路径
// 见 shortest_path_binary_matrix.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

int shortestPathBinaryMatrix(const std::vector<std::vector<int>> &grid) {
    int n = grid.size();
    if (grid[0][0] == 1 || grid[n - 1][n - 1] == 1) {
        return -1;
    }
    std::vector<std::vector<int>> dist(n, std::vector<int>(n, 0));
    dist[0][0] = 1;
    std::queue<std::pair<int, int>> q;
    q.push({0, 0});

    const int dr[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dc[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    while (!q.empty()) {
        auto [r, c] = q.front();
        q.pop();
        if (r == n - 1 && c == n - 1) {
            return dist[r][c];
        }
        for (int d = 0; d < 8; ++d) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr >= 0 && nr < n && nc >= 0 && nc < n && grid[nr][nc] == 0 &&
                dist[nr][nc] == 0) {
                dist[nr][nc] = dist[r][c] + 1;
                q.push({nr, nc});
            }
        }
    }
    return -1;
}

int main() {
    std::vector<std::vector<int>> g1 = {{0, 1}, {1, 0}};
    assert(shortestPathBinaryMatrix(g1) == 2);

    std::vector<std::vector<int>> g2 = {{0, 0, 0}, {1, 1, 0}, {1, 1, 0}};
    assert(shortestPathBinaryMatrix(g2) == 4);

    std::vector<std::vector<int>> g3 = {{1, 0, 0}, {1, 1, 0}, {1, 1, 0}};
    assert(shortestPathBinaryMatrix(g3) == -1);

    std::vector<std::vector<int>> g4 = {{0}};
    assert(shortestPathBinaryMatrix(g4) == 1);

    std::vector<std::vector<int>> g5 = {{1}};
    assert(shortestPathBinaryMatrix(g5) == -1);

    std::vector<std::vector<int>> g6 = {{0, 0}, {0, 0}};
    assert(shortestPathBinaryMatrix(g6) == 2);

    std::cout << "shortest_path_binary_matrix: all tests passed\n";
    return 0;
}
