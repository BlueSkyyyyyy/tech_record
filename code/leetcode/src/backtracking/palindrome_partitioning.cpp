// 131. 分割回文串
// 见 palindrome_partitioning.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

bool isPalindrome(const std::string &s, int lo, int hi) {
    while (lo < hi) {
        if (s[lo] != s[hi]) return false;
        ++lo;
        --hi;
    }
    return true;
}

void backtrack(const std::string &s, int start, std::vector<std::string> &path,
               std::vector<std::vector<std::string>> &res) {
    if (start == static_cast<int>(s.size())) {
        res.push_back(path);
        return;
    }
    for (int end = start + 1; end <= static_cast<int>(s.size()); ++end) {
        if (!isPalindrome(s, start, end - 1)) continue;
        path.push_back(s.substr(start, end - start));
        backtrack(s, end, path, res);
        path.pop_back();
    }
}

std::vector<std::vector<std::string>> partition(const std::string &s) {
    std::vector<std::vector<std::string>> res;
    std::vector<std::string> path;
    backtrack(s, 0, path, res);
    return res;
}

int main() {
    auto got = partition("aab");
    std::vector<std::vector<std::string>> want = {{"a", "a", "b"}, {"aa", "b"}};
    assert(got == want);

    auto got2 = partition("a");
    std::vector<std::vector<std::string>> want2 = {{"a"}};
    assert(got2 == want2);

    auto got3 = partition("aba");
    std::vector<std::vector<std::string>> want3 = {{"a", "b", "a"}, {"aba"}};
    assert(got3 == want3);

    std::cout << "palindrome_partitioning: all tests passed\n";
    return 0;
}
