"""1823. 找出游戏的获胜者（Find the Winner of the Circular Game）

题目：n 个朋友围成一圈，编号 1..n。从 1 号开始报数，每报到 k 的人出局，
出局者的下一位从 1 重新报数。重复到只剩一人，返回获胜者编号。

思路（队列模拟真实的报数流程）：
    把 1..n 依次放进队列。每次淘汰一个人之前，先把队首的 k-1 个人依次
    「出队再入队」——他们报的是 1..k-1，这一轮安全，转到队尾等待下一轮；
    此时队首正好是报到 k 的人，直接弹出即淘汰。重复 n-1 次，队里只剩获胜者。

    队列的 FIFO 语义天然对应「一圈人依次报数、淘汰后从下一位继续」，
    不用手工维护下标绕圈。

复杂度：时间 O(n*k)，空间 O(n)。（有 O(n) 的约瑟夫递推公式，见相似题。）
"""

from collections import deque


def find_the_winner(n, k):
    q = deque(range(1, n + 1))
    while len(q) > 1:
        for _ in range(k - 1):
            q.append(q.popleft())
        q.popleft()
    return q[0]


if __name__ == "__main__":
    assert find_the_winner(5, 2) == 3
    assert find_the_winner(6, 5) == 1
    assert find_the_winner(1, 1) == 1
    assert find_the_winner(2, 2) == 1
    assert find_the_winner(5, 1) == 5
    print("find_the_winner: all tests passed")
