// 433. 最小基因变化
// 见 min_genetic_mutation.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

int minGeneticMutation(const std::string &startGene, const std::string &endGene,
                       const std::vector<std::string> &bank) {
    std::unordered_set<std::string> genes(bank.begin(), bank.end());
    if (genes.find(endGene) == genes.end()) {
        return -1;
    }
    if (startGene == endGene) {
        return 0;
    }
    const std::string bases = "ACGT";
    std::queue<std::string> q;
    std::unordered_set<std::string> visited;
    q.push(startGene);
    visited.insert(startGene);
    int steps = 0;
    while (!q.empty()) {
        ++steps;
        int level = q.size();
        for (int i = 0; i < level; ++i) {
            std::string cur = q.front();
            q.pop();
            for (int p = 0; p < static_cast<int>(cur.size()); ++p) {
                char old = cur[p];
                for (char g : bases) {
                    if (g == old) {
                        continue;
                    }
                    cur[p] = g;
                    if (genes.count(cur) && !visited.count(cur)) {
                        if (cur == endGene) {
                            return steps;
                        }
                        visited.insert(cur);
                        q.push(cur);
                    }
                }
                cur[p] = old;
            }
        }
    }
    return -1;
}

int main() {
    assert(minGeneticMutation("AACCGGTT", "AACCGGTA", {"AACCGGTA"}) == 1);
    assert(minGeneticMutation("AACCGGTT", "AAACGGTA",
                              {"AACCGGTA", "AACCGCTA", "AAACGGTA"}) == 2);
    assert(minGeneticMutation("AAAAACCC", "AACCCCCC",
                              {"AAAACCCC", "AAACCCCC", "AACCCCCC"}) == 3);
    assert(minGeneticMutation("AACCGGTT", "AACCGGTA", {}) == -1);

    std::cout << "min_genetic_mutation: all tests passed\n";
    return 0;
}
