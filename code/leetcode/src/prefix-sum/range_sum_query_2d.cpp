// 304. 二维区域和检索 - 矩阵不可变（二维前缀和 + 容斥）
// 见 range_sum_query_2d.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

class NumMatrix {
public:
    explicit NumMatrix(const std::vector<std::vector<int>> &matrix) {
        int m = static_cast<int>(matrix.size());
        int n = m ? static_cast<int>(matrix[0].size()) : 0;
        prefix_.assign(m + 1, std::vector<long long>(n + 1, 0));
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                prefix_[i + 1][j + 1] = matrix[i][j] + prefix_[i][j + 1] +
                                        prefix_[i + 1][j] - prefix_[i][j];
    }

    int sumRegion(int row1, int col1, int row2, int col2) const {
        return static_cast<int>(prefix_[row2 + 1][col2 + 1] -
                                prefix_[row1][col2 + 1] -
                                prefix_[row2 + 1][col1] + prefix_[row1][col1]);
    }

private:
    std::vector<std::vector<long long>> prefix_;
};

int main() {
    NumMatrix nm({
        {3, 0, 1, 4, 2},
        {5, 6, 3, 2, 1},
        {1, 2, 0, 1, 5},
        {4, 1, 0, 1, 7},
        {1, 0, 3, 0, 5},
    });
    assert(nm.sumRegion(2, 1, 4, 3) == 8);
    assert(nm.sumRegion(1, 1, 2, 2) == 11);
    assert(nm.sumRegion(0, 0, 0, 0) == 3);
    assert(nm.sumRegion(4, 4, 4, 4) == 5);
    assert(nm.sumRegion(0, 0, 4, 4) == 58);
    std::vector<std::vector<int>> single_matrix = {{7}};
    NumMatrix single(single_matrix);
    assert(single.sumRegion(0, 0, 0, 0) == 7);
    std::cout << "range_sum_query_2d: all tests passed\n";
    return 0;
}
