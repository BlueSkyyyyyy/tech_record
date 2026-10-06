// 1255. 得分最高的单词集合
// 见 maximum_score_words_formed.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

int maxScoreWords(std::vector<std::string> &words, std::vector<char> &letters,
                  std::vector<int> &score) {
    int have[26] = {0};
    for (char ch : letters) {
        have[ch - 'a']++;
    }

    int n = static_cast<int>(words.size());
    std::vector<std::vector<int>> wordCount(n, std::vector<int>(26, 0));
    std::vector<int> wordScore(n, 0);
    for (int i = 0; i < n; ++i) {
        bool ok = true;
        for (char ch : words[i]) {
            int idx = ch - 'a';
            wordCount[i][idx]++;
            if (wordCount[i][idx] > have[idx]) {
                ok = false;
            }
            wordScore[i] += score[idx];
        }
        if (!ok) {
            wordScore[i] = 0;
        }
    }

    int best = 0;
    for (int mask = 0; mask < (1 << n); ++mask) {
        int used[26] = {0};
        int total = 0;
        bool ok = true;
        for (int i = 0; i < n && ok; ++i) {
            if (mask >> i & 1) {
                for (int k = 0; k < 26; ++k) {
                    used[k] += wordCount[i][k];
                    if (used[k] > have[k]) {
                        ok = false;
                        break;
                    }
                }
                total += wordScore[i];
            }
        }
        if (ok) {
            best = std::max(best, total);
        }
    }
    return best;
}

int main() {
    std::vector<std::string> w1 = {"dog", "cat", "dad", "good"};
    std::vector<char> l1 = {'a', 'a', 'c', 'd', 'd', 'd', 'g', 'o', 'o'};
    std::vector<int> s1 = {1, 0, 9, 5, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    assert(maxScoreWords(w1, l1, s1) == 23);

    std::vector<std::string> w2 = {"a", "b", "c"};
    std::vector<char> l2 = {'a', 'a', 'b', 'c'};
    std::vector<int> s2 = {1, 10, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    assert(maxScoreWords(w2, l2, s2) == 111);

    std::vector<std::string> w4 = {"a", "aa", "aaa"};
    std::vector<char> l4 = {'a', 'a', 'a', 'a'};
    assert(maxScoreWords(w4, l4, s2) == 4);

    std::vector<std::string> w3 = {"leetcode"};
    std::vector<char> l3 = {'l', 'e', 't', 'c', 'o', 'd'};
    std::vector<int> s3(26, 0);
    assert(maxScoreWords(w3, l3, s3) == 0);

    std::cout << "maximum_score_words_formed: all tests passed\n";
    return 0;
}
