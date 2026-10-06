// 77. 组合
// 见 combinations.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

void backtrack(int n, int k, int start, std::vector<int> &path,
               std::vector<std::vector<int>> &res) {
    if (static_cast<int>(path.size()) == k) {
        res.push_back(path);
        return;
    }
    int need = k - static_cast<int>(path.size());
    for (int i = start; i <= n - need + 1; ++i) {
        path.push_back(i);
        backtrack(n, k, i + 1, path, res);
        path.pop_back();
    }
}

std::vector<std::vector<int>> combine(int n, int k) {
    std::vector<std::vector<int>> res;
    std::vector<int> path;
    backtrack(n, k, 1, path, res);
    return res;
}

int main() {
    auto got = combine(4, 2);
    std::vector<std::vector<int>> want = {
        {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}};
    assert(got == want);

    auto got2 = combine(1, 1);
    std::vector<std::vector<int>> want2 = {{1}};
    assert(got2 == want2);

    auto got3 = combine(3, 3);
    std::vector<std::vector<int>> want3 = {{1, 2, 3}};
    assert(got3 == want3);

    auto got4 = combine(3, 4);
    assert(got4.empty());

    std::cout << "combinations: all tests passed\n";
    return 0;
}
