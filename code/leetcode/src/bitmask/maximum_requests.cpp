// 1601. 最多可达成的换楼请求数目
// 见 maximum_requests.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int maximumRequests(int n, std::vector<std::vector<int>> &requests) {
    int R = static_cast<int>(requests.size());
    int best = 0;
    for (int mask = 0; mask < (1 << R); ++mask) {
        std::vector<int> delta(n, 0);
        int count = 0;
        for (int i = 0; i < R; ++i) {
            if (mask >> i & 1) {
                delta[requests[i][0]]--;
                delta[requests[i][1]]++;
                count++;
            }
        }
        bool ok = true;
        for (int d : delta) {
            if (d != 0) {
                ok = false;
                break;
            }
        }
        if (ok && count > best) {
            best = count;
        }
    }
    return best;
}

int main() {
    {
        std::vector<std::vector<int>> req = {{0, 1}, {1, 0}, {0, 1}, {1, 2}, {2, 0}, {3, 4}};
        assert(maximumRequests(5, req) == 5);
    }
    {
        std::vector<std::vector<int>> req = {{0, 0}, {1, 2}, {2, 1}};
        assert(maximumRequests(3, req) == 3);
    }
    {
        std::vector<std::vector<int>> req = {{0, 3}, {3, 1}, {1, 2}, {2, 0}};
        assert(maximumRequests(4, req) == 4);
    }
    {
        std::vector<std::vector<int>> req = {{0, 1}};
        assert(maximumRequests(2, req) == 0);
    }

    std::cout << "maximum_requests: all tests passed\n";
    return 0;
}
