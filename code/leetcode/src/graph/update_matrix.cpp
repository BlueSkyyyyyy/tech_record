// 542. 01 矩阵
// 见 update_matrix.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

std::vector<std::vector<int>> updateMatrix(const std::vector<std::vector<int>> &mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    std::vector<std::vector<int>> dist(rows, std::vector<int>(cols, -1));
    std::queue<std::pair<int, int>> q;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (mat[r][c] == 0) {
                dist[r][c] = 0;
                q.push({r, c});
            }
        }
    }

    const int dr[4] = {1, -1, 0, 0};
    const int dc[4] = {0, 0, 1, -1};
    while (!q.empty()) {
        auto [r, c] = q.front();
        q.pop();
        for (int d = 0; d < 4; ++d) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols &&
                dist[nr][nc] == -1) {
                dist[nr][nc] = dist[r][c] + 1;
                q.push({nr, nc});
            }
        }
    }
    return dist;
}

int main() {
    std::vector<std::vector<int>> m1 = {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}};
    std::vector<std::vector<int>> want1 = {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}};
    assert(updateMatrix(m1) == want1);

    std::vector<std::vector<int>> m2 = {{0, 0, 0}, {0, 1, 0}, {1, 1, 1}};
    std::vector<std::vector<int>> want2 = {{0, 0, 0}, {0, 1, 0}, {1, 2, 1}};
    assert(updateMatrix(m2) == want2);

    std::vector<std::vector<int>> m3 = {{0}};
    std::vector<std::vector<int>> want3 = {{0}};
    assert(updateMatrix(m3) == want3);

    std::vector<std::vector<int>> m4 = {{1, 1}, {1, 1}};
    std::vector<std::vector<int>> want4 = {{-1, -1}, {-1, -1}};
    assert(updateMatrix(m4) == want4);

    std::cout << "update_matrix: all tests passed\n";
    return 0;
}
