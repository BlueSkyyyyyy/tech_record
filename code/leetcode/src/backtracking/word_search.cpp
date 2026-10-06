// 79. 单词搜索
// 见 word_search.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

bool dfs(std::vector<std::string> &board, const std::string &word, int r,
         int c, int k) {
    if (k == static_cast<int>(word.size())) return true;
    int rows = board.size(), cols = board[0].size();
    if (r < 0 || r >= rows || c < 0 || c >= cols || board[r][c] != word[k])
        return false;
    char saved = board[r][c];
    board[r][c] = '#';
    bool found = dfs(board, word, r + 1, c, k + 1) ||
                 dfs(board, word, r - 1, c, k + 1) ||
                 dfs(board, word, r, c + 1, k + 1) ||
                 dfs(board, word, r, c - 1, k + 1);
    board[r][c] = saved;
    return found;
}

bool exist(std::vector<std::string> board, const std::string &word) {
    int rows = board.size(), cols = board[0].size();
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (dfs(board, word, r, c, 0)) return true;
        }
    }
    return false;
}

int main() {
    std::vector<std::string> board = {"ABCE", "SFCS", "ADEE"};
    assert(exist(board, "ABCCED") == true);
    assert(exist(board, "SEE") == true);
    assert(exist(board, "ABCB") == false);

    assert(exist({"a"}, "a") == true);
    assert(exist({"a"}, "b") == false);

    std::cout << "word_search: all tests passed\n";
    return 0;
}
