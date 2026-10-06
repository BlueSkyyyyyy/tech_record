// 36. 有效的数独
// 见 valid_sudoku.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

bool isValidSudoku(const std::vector<std::string> &board) {
    std::vector<std::unordered_set<char>> rows(9), cols(9), boxes(9);
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            char val = board[i][j];
            if (val == '.') {
                continue;
            }
            int k = (i / 3) * 3 + j / 3;
            if (rows[i].count(val) || cols[j].count(val) || boxes[k].count(val)) {
                return false;
            }
            rows[i].insert(val);
            cols[j].insert(val);
            boxes[k].insert(val);
        }
    }
    return true;
}

int main() {
    std::vector<std::string> valid = {
        "53..7....", "6..195...", ".98....6.", "8...6...3", "4..8.3..1",
        "7...2...6", ".6....28.", "...419..5", "....8..79"};
    assert(isValidSudoku(valid));

    std::vector<std::string> invalid = {
        "83..7....", "6..195...", ".98....6.", "8...6...3", "4..8.3..1",
        "7...2...6", ".6....28.", "...419..5", "....8..79"};
    assert(!isValidSudoku(invalid));

    std::vector<std::string> empty(9, ".........");
    assert(isValidSudoku(empty));

    std::cout << "is_valid_sudoku: all tests passed\n";
    return 0;
}
