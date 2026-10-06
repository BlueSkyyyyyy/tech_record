// 1125. 最小的必要团队
// 见 smallest_sufficient_team.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

std::vector<int> smallestSufficientTeam(std::vector<std::string> &reqSkills,
                                        std::vector<std::vector<std::string>> &people) {
    int m = static_cast<int>(reqSkills.size());
    std::unordered_map<std::string, int> skillId;
    for (int i = 0; i < m; ++i) {
        skillId[reqSkills[i]] = i;
    }

    int n = static_cast<int>(people.size());
    std::vector<int> peopleMask(n, 0);
    for (int i = 0; i < n; ++i) {
        for (const std::string &skill : people[i]) {
            peopleMask[i] |= 1 << skillId[skill];
        }
    }

    int full = (1 << m) - 1;
    std::vector<std::vector<int>> dp(1 << m);
    std::vector<char> reachable(1 << m, 0);
    dp[0] = {};
    reachable[0] = 1;
    for (int i = 0; i < n; ++i) {
        std::vector<int> masks;
        for (int k = 0; k < (1 << m); ++k) {
            if (reachable[k]) {
                masks.push_back(k);
            }
        }
        for (int covered : masks) {
            int merged = covered | peopleMask[i];
            if (merged == covered) {
                continue;
            }
            std::vector<int> cand = dp[covered];
            cand.push_back(i);
            if (!reachable[merged] || dp[merged].size() > cand.size()) {
                dp[merged] = cand;
                reachable[merged] = 1;
            }
        }
    }
    return dp[full];
}

int main() {
    {
        std::vector<std::string> req = {"java", "nodejs", "reactjs"};
        std::vector<std::vector<std::string>> p = {{"java"}, {"nodejs"}, {"nodejs", "reactjs"}};
        std::vector<int> want = {0, 2};
        assert(smallestSufficientTeam(req, p) == want);
    }
    {
        std::vector<std::string> req = {"algorithms", "math", "java", "reactjs", "csharp", "aws"};
        std::vector<std::vector<std::string>> p = {
            {"algorithms", "math", "java"},
            {"algorithms", "math", "reactjs"},
            {"java", "csharp", "aws"},
            {"reactjs", "csharp"},
            {"csharp", "math"},
            {"aws", "java"}};
        auto got = smallestSufficientTeam(req, p);
        std::sort(got.begin(), got.end());
        std::vector<int> want = {1, 2};
        assert(got == want);
    }
    {
        std::vector<std::string> req = {"c", "cpp"};
        std::vector<std::vector<std::string>> p = {{"c", "cpp"}};
        std::vector<int> want = {0};
        assert(smallestSufficientTeam(req, p) == want);
    }

    std::cout << "smallest_sufficient_team: all tests passed\n";
    return 0;
}
