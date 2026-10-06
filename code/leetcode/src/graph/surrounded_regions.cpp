// 130. 被围绕的区域
// 见 surrounded_regions.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

void dfs(std::vector<std::vector<char>> &board, int r, int c) {
    int rows = board.size();
    int cols = board[0].size();
    if (r < 0 || r >= rows || c < 0 || c >= cols || board[r][c] != 'O') return;
    board[r][c] = '#';
    dfs(board, r + 1, c);
    dfs(board, r - 1, c);
    dfs(board, r, c + 1);
    dfs(board, r, c - 1);
}

void solve(std::vector<std::vector<char>> &board) {
    if (board.empty() || board[0].empty()) return;
    int rows = board.size();
    int cols = board[0].size();

    for (int r = 0; r < rows; ++r) {
        dfs(board, r, 0);
        dfs(board, r, cols - 1);
    }
    for (int c = 0; c < cols; ++c) {
        dfs(board, 0, c);
        dfs(board, rows - 1, c);
    }

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (board[r][c] == 'O') {
                board[r][c] = 'X';
            } else if (board[r][c] == '#') {
                board[r][c] = 'O';
            }
        }
    }
}

int main() {
    std::vector<std::vector<char>> b1 = {
        {'X', 'X', 'X', 'X'},
        {'X', 'O', 'O', 'X'},
        {'X', 'X', 'O', 'X'},
        {'X', 'O', 'X', 'X'},
    };
    solve(b1);
    std::vector<std::vector<char>> want1 = {
        {'X', 'X', 'X', 'X'},
        {'X', 'X', 'X', 'X'},
        {'X', 'X', 'X', 'X'},
        {'X', 'O', 'X', 'X'},
    };
    assert(b1 == want1);

    std::vector<std::vector<char>> b2 = {{'O', 'O'}, {'O', 'O'}};
    solve(b2);
    std::vector<std::vector<char>> want2 = {{'O', 'O'}, {'O', 'O'}};
    assert(b2 == want2);

    std::vector<std::vector<char>> b3 = {{'X'}};
    solve(b3);
    std::vector<std::vector<char>> want3 = {{'X'}};
    assert(b3 == want3);

    std::cout << "surrounded_regions: all tests passed\n";
    return 0;
}
