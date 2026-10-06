// 210. 课程表 II
// 见 course_schedule_ii.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <queue>
#include <vector>

std::vector<int> findOrder(int numCourses,
                           std::vector<std::vector<int>> &prerequisites) {
    std::vector<std::vector<int>> graph(numCourses);
    std::vector<int> indegree(numCourses, 0);
    for (auto &p : prerequisites) {
        int course = p[0], pre = p[1];
        graph[pre].push_back(course);
        ++indegree[course];
    }

    std::queue<int> q;
    for (int c = 0; c < numCourses; ++c) {
        if (indegree[c] == 0) q.push(c);
    }

    std::vector<int> order;
    while (!q.empty()) {
        int cur = q.front();
        q.pop();
        order.push_back(cur);
        for (int nxt : graph[cur]) {
            if (--indegree[nxt] == 0) q.push(nxt);
        }
    }

    if (static_cast<int>(order.size()) != numCourses) return {};
    return order;
}

int main() {
    std::vector<std::vector<int>> p1 = {{1, 0}};
    std::vector<int> want1 = {0, 1};
    assert(findOrder(2, p1) == want1);

    std::vector<std::vector<int>> p2 = {{1, 0}, {2, 0}, {3, 1}, {3, 2}};
    std::vector<int> want2 = {0, 1, 2, 3};
    assert(findOrder(4, p2) == want2);

    std::vector<std::vector<int>> none;
    std::vector<int> want3 = {0};
    assert(findOrder(1, none) == want3);

    std::vector<std::vector<int>> cycle = {{1, 0}, {0, 1}};
    assert(findOrder(2, cycle).empty());

    std::cout << "course_schedule_ii: all tests passed\n";
    return 0;
}
