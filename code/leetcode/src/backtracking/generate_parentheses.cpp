// 22. 括号生成
// 见 generate_parentheses.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

void backtrack(int n, std::string &cur, int open_, int close,
               std::vector<std::string> &res) {
    if (static_cast<int>(cur.size()) == 2 * n) {
        res.push_back(cur);
        return;
    }
    if (open_ < n) {
        cur.push_back('(');
        backtrack(n, cur, open_ + 1, close, res);
        cur.pop_back();
    }
    if (close < open_) {
        cur.push_back(')');
        backtrack(n, cur, open_, close + 1, res);
        cur.pop_back();
    }
}

std::vector<std::string> generateParenthesis(int n) {
    std::vector<std::string> res;
    std::string cur;
    backtrack(n, cur, 0, 0, res);
    return res;
}

int main() {
    auto got = generateParenthesis(3);
    assert(got.size() == 5);
    auto has = [&](const std::string &v) {
        for (auto &x : got)
            if (x == v) return true;
        return false;
    };
    assert(has("((()))") && has("()()()") && !has("())("));

    auto got1 = generateParenthesis(1);
    std::vector<std::string> want1 = {"()"};
    assert(got1 == want1);

    auto got0 = generateParenthesis(0);
    std::vector<std::string> want0 = {""};
    assert(got0 == want0);

    std::cout << "generate_parentheses: all tests passed\n";
    return 0;
}
