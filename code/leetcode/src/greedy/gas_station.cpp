// 134. 加油站
// 见 gas_station.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int canCompleteCircuit(const std::vector<int> &gas, const std::vector<int> &cost) {
    int total = 0;
    int tank = 0;
    int start = 0;
    int n = static_cast<int>(gas.size());
    for (int i = 0; i < n; ++i) {
        int diff = gas[i] - cost[i];
        total += diff;
        tank += diff;
        if (tank < 0) {
            start = i + 1;
            tank = 0;
        }
    }
    return total >= 0 ? start : -1;
}

int main() {
    std::vector<int> g1 = {1, 2, 3, 4, 5}, c1 = {3, 4, 5, 1, 2};
    assert(canCompleteCircuit(g1, c1) == 3);
    std::vector<int> g2 = {2, 3, 4}, c2 = {3, 4, 3};
    assert(canCompleteCircuit(g2, c2) == -1);
    std::vector<int> g3 = {5}, c3 = {4};
    assert(canCompleteCircuit(g3, c3) == 0);
    std::vector<int> g4 = {2}, c4 = {3};
    assert(canCompleteCircuit(g4, c4) == -1);
    std::vector<int> g5 = {3, 1, 1}, c5 = {1, 2, 2};
    assert(canCompleteCircuit(g5, c5) == 0);
    std::cout << "gas_station: all tests passed\n";
    return 0;
}
