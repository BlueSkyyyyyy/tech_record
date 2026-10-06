// 118. 杨辉三角
// 见 pascals_triangle.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

std::vector<std::vector<int>> generate(int numRows) {
    std::vector<std::vector<int>> triangle;
    for (int i = 0; i < numRows; ++i) {
        std::vector<int> row(i + 1, 1);
        for (int j = 1; j < i; ++j) {
            row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j];
        }
        triangle.push_back(row);
    }
    return triangle;
}

int main() {
    std::vector<std::vector<int>> want1 = {{1}};
    std::vector<std::vector<int>> want2 = {{1}, {1, 1}};
    std::vector<std::vector<int>> want5 = {
        {1}, {1, 1}, {1, 2, 1}, {1, 3, 3, 1}, {1, 4, 6, 4, 1}};
    assert(generate(0).empty());
    assert(generate(1) == want1);
    assert(generate(2) == want2);
    assert(generate(5) == want5);
    std::cout << "pascals_triangle: all tests passed\n";
    return 0;
}
