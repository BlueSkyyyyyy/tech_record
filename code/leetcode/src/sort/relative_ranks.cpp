// 506. 相对名次
// 见 relative_ranks.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

std::vector<std::string> findRelativeRanks(const std::vector<int> &score) {
    int n = static_cast<int>(score.size());
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&score](int a, int b) { return score[a] > score[b]; });
    std::vector<std::string> medals = {"Gold Medal", "Silver Medal", "Bronze Medal"};
    std::vector<std::string> res(n);
    for (int rank = 0; rank < n; ++rank) {
        int i = order[rank];
        res[i] = rank < 3 ? medals[rank] : std::to_string(rank + 1);
    }
    return res;
}

int main() {
    std::vector<std::string> want1 = {"Gold Medal", "Silver Medal", "Bronze Medal",
                                      "4", "5"};
    assert(findRelativeRanks({5, 4, 3, 2, 1}) == want1);

    std::vector<std::string> want2 = {"Gold Medal", "5", "Bronze Medal",
                                      "Silver Medal", "4"};
    assert(findRelativeRanks({10, 3, 8, 9, 4}) == want2);

    std::vector<std::string> want3 = {"Gold Medal"};
    assert(findRelativeRanks({1}) == want3);

    std::vector<std::string> want4 = {"Gold Medal", "Bronze Medal", "Silver Medal"};
    assert(findRelativeRanks({3, 1, 2}) == want4);

    std::cout << "relative_ranks: all tests passed\n";
    return 0;
}
