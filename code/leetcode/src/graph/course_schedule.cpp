// 207. 课程表
// 见 course_schedule.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <queue>
#include <vector>

bool canFinish(int numCourses, std::vector<std::vector<int>> &prerequisites) {
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

    int done = 0;
    while (!q.empty()) {
        int cur = q.front();
        q.pop();
        ++done;
        for (int nxt : graph[cur]) {
            if (--indegree[nxt] == 0) q.push(nxt);
        }
    }
    return done == numCourses;
}

int main() {
    std::vector<std::vector<int>> p1 = {{1, 0}};
    assert(canFinish(2, p1) == true);

    std::vector<std::vector<int>> p2 = {{1, 0}, {0, 1}};
    assert(canFinish(2, p2) == false);

    std::vector<std::vector<int>> none;
    assert(canFinish(1, none) == true);

    std::vector<std::vector<int>> p3 = {{1, 4}, {2, 4}, {3, 1}, {3, 2}};
    assert(canFinish(5, p3) == true);

    std::vector<std::vector<int>> self = {{0, 0}};
    assert(canFinish(1, self) == false);

    std::vector<std::vector<int>> cycle = {{1, 0}, {2, 1}, {0, 2}};
    assert(canFinish(3, cycle) == false);

    std::cout << "course_schedule: all tests passed\n";
    return 0;
}
