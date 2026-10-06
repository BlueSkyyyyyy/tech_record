// 187. 重复的 DNA 序列
// 见 repeated_dna_sequences.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

std::vector<std::string> findRepeatedDnaSequences(const std::string &s) {
    const int length = 10;
    std::vector<std::string> res;
    if (static_cast<int>(s.size()) < length + 1) {
        return res;
    }
    std::unordered_set<std::string> seen, added;
    for (int i = 0; i + length <= static_cast<int>(s.size()); ++i) {
        std::string sub = s.substr(i, length);
        if (seen.count(sub) && !added.count(sub)) {
            res.push_back(sub);
            added.insert(sub);
        }
        seen.insert(sub);
    }
    return res;
}

int main() {
    std::vector<std::string> want = {"AAAAACCCCC", "CCCCCAAAAA"};
    assert(findRepeatedDnaSequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT") == want);
    std::vector<std::string> one = {"AAAAAAAAAA"};
    assert(findRepeatedDnaSequences("AAAAAAAAAAAAA") == one);
    assert(findRepeatedDnaSequences("ACGT").empty());
    assert(findRepeatedDnaSequences("").empty());

    std::cout << "findRepeatedDnaSequences: all tests passed\n";
    return 0;
}
