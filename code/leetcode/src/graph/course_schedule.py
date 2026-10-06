"""207. 课程表（Course Schedule）

题目：你需要选修 numCourses 门课，编号 0 ~ numCourses-1。
先修关系用 prerequisites 表示，其中 prerequisites[i] = [a, b] 表示
修课程 a 之前必须先修课程 b。判断是否可能修完所有课程。

思路（拓扑排序 / Kahn 算法）：
    把课程看作有向图的节点，先修关系 b -> a 看作一条有向边。
    「能否修完」等价于「这张有向图有没有环」：有环说明几门课互为先修，无法开始。
    若把「已解锁、可以修」的课程看作入度为 0 的节点，就得到一个自然的流程：

    1. 统计每门课的入度（有多少门先修课还没修），并把所有入度为 0 的课入队；
    2. 反复取出队首的课，把它「修掉」（计数加一），
       并把它指向的后续课程的入度减一；某门课入度减到 0 就入队；
    3. 最后看修掉的课程数是否等于总数。相等就说明无环、能修完。

    为什么入度为 0 代表可以修：所有指向它的前置课程都已经修完，没有未满足的约束。
    为什么有环就修不完：环上的每门课都在等环上的另一门，入度永远降不到 0，永远出不了队。

复杂度：时间 O(V + E)（建图 + 每条边处理一次），空间 O(V + E)（邻接表 + 队列）。
"""
from collections import deque


def can_finish(num_courses, prerequisites):
    graph = [[] for _ in range(num_courses)]
    indegree = [0] * num_courses
    for course, pre in prerequisites:
        graph[pre].append(course)
        indegree[course] += 1

    queue = deque(c for c in range(num_courses) if indegree[c] == 0)
    done = 0
    while queue:
        cur = queue.popleft()
        done += 1
        for nxt in graph[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return done == num_courses


if __name__ == "__main__":
    assert can_finish(2, [[1, 0]]) is True
    assert can_finish(2, [[1, 0], [0, 1]]) is False
    assert can_finish(1, []) is True
    assert can_finish(5, [[1, 4], [2, 4], [3, 1], [3, 2]]) is True
    # 自环：课程 0 以自己为先修
    assert can_finish(1, [[0, 0]]) is False
    assert can_finish(3, [[1, 0], [2, 1], [0, 2]]) is False
    print("course_schedule: all tests passed")
