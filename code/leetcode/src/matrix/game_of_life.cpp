// 289. 生命游戏
// 见 game_of_life.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

void gameOfLife(std::vector<std::vector<int>> &board) {
    int m = static_cast<int>(board.size());
    int n = static_cast<int>(board[0].size());
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            int live = 0;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    if (di == 0 && dj == 0) {
                        continue;
                    }
                    int ni = i + di;
                    int nj = j + dj;
                    if (ni >= 0 && ni < m && nj >= 0 && nj < n) {
                        live += board[ni][nj] & 1;
                    }
                }
            }
            if (board[i][j] & 1) {
                board[i][j] = (live == 2 || live == 3) ? 0b11 : 0b01;
            } else {
                board[i][j] = (live == 3) ? 0b10 : 0b00;
            }
        }
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            board[i][j] >>= 1;
        }
    }
}

int main() {
    std::vector<std::vector<int>> b1 = {{0, 1, 0}, {0, 0, 1}, {1, 1, 1}, {0, 0, 0}};
    std::vector<std::vector<int>> want1 = {{0, 0, 0}, {1, 0, 1}, {0, 1, 1}, {0, 1, 0}};
    gameOfLife(b1);
    assert(b1 == want1);

    std::vector<std::vector<int>> b2 = {{1, 1}, {1, 0}};
    std::vector<std::vector<int>> want2 = {{1, 1}, {1, 1}};
    gameOfLife(b2);
    assert(b2 == want2);

    std::vector<std::vector<int>> b3 = {{0, 0}, {0, 0}};
    std::vector<std::vector<int>> want3 = {{0, 0}, {0, 0}};
    gameOfLife(b3);
    assert(b3 == want3);

    std::vector<std::vector<int>> b4 = {{1}};
    std::vector<std::vector<int>> want4 = {{0}};
    gameOfLife(b4);
    assert(b4 == want4);

    std::cout << "game_of_life: all tests passed\n";
    return 0;
}
