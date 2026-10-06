// 139. 单词拆分
// 见 word_break.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

bool wordBreak(const std::string &s, const std::vector<std::string> &wordDict) {
    std::unordered_set<std::string> words(wordDict.begin(), wordDict.end());
    int n = static_cast<int>(s.size());
    std::vector<bool> dp(n + 1, false);
    dp[0] = true;
    for (int i = 1; i <= n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (dp[j] && words.count(s.substr(j, i - j))) {
                dp[i] = true;
                break;
            }
        }
    }
    return dp[n];
}

int main() {
    std::vector<std::string> a = {"leet", "code"};
    std::vector<std::string> b = {"apple", "pen"};
    std::vector<std::string> c = {"cats", "dog", "sand", "and", "cat"};
    std::vector<std::string> d = {};
    std::vector<std::string> e = {"aaaa", "aaa"};
    assert(wordBreak("leetcode", a) == true);
    assert(wordBreak("applepenapple", b) == true);
    assert(wordBreak("catsandog", c) == false);
    assert(wordBreak("", d) == true);
    assert(wordBreak("a", d) == false);
    assert(wordBreak("aaaaaaa", e) == true);
    std::cout << "word_break: all tests passed\n";
    return 0;
}
