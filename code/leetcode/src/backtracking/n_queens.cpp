// 51. N 皇后
// 见 n_queens.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

void backtrack(int n, int r, std::vector<std::string> &board,
               std::unordered_set<int> &cols, std::unordered_set<int> &diagMain,
               std::unordered_set<int> &diagAnti,
               std::vector<std::vector<std::string>> &res) {
    if (r == n) {
        res.push_back(board);
        return;
    }
    for (int c = 0; c < n; ++c) {
        if (cols.count(c) || diagMain.count(r - c) || diagAnti.count(r + c))
            continue;
        cols.insert(c);
        diagMain.insert(r - c);
        diagAnti.insert(r + c);
        board[r][c] = 'Q';
        backtrack(n, r + 1, board, cols, diagMain, diagAnti, res);
        board[r][c] = '.';
        cols.erase(c);
        diagMain.erase(r - c);
        diagAnti.erase(r + c);
    }
}

std::vector<std::vector<std::string>> solveNQueens(int n) {
    std::vector<std::vector<std::string>> res;
    std::vector<std::string> board(n, std::string(n, '.'));
    std::unordered_set<int> cols, diagMain, diagAnti;
    backtrack(n, 0, board, cols, diagMain, diagAnti, res);
    return res;
}

int main() {
    auto got4 = solveNQueens(4);
    assert(got4.size() == 2);
    std::vector<std::string> a = {".Q..", "...Q", "Q...", "..Q."};
    std::vector<std::string> b = {"..Q.", "Q...", "...Q", ".Q.."};
    for (auto &want : {a, b}) {
        assert(std::find(got4.begin(), got4.end(), want) != got4.end());
    }

    assert(solveNQueens(1).size() == 1);
    assert(solveNQueens(2).empty());
    assert(solveNQueens(8).size() == 92);

    std::cout << "n_queens: all tests passed\n";
    return 0;
}
