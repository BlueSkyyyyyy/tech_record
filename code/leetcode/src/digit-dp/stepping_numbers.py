"""1215. 步进数（Stepping Numbers）

题目：步进数是相邻两位数字都正好相差 1 的数。给定 low、high，按升序返回
[low, high] 内所有步进数。

思路（按位生长，而不是数位 DP）：
    本题要的是**具体的数**而不是个数，所以更适合「生成」：从一个一位数开始，
    每次在末尾接一个与末位相差 1 的数字，就得到一个新的步进数。

    用队列/BFS 做这种「逐位生长」：初始把 1..9 入队（0 单独判断是否在范围内）。
    每次取队首 x：
      - 若 x > high，它和它的后代都超标，跳过；
      - 若 x >= low，收进答案；
      - 末位是 last，则可以长出 x*10 + (last-1)（last>0 时）和
        x*10 + (last+1)（last<9 时）两个后代，入队。
    因为生长顺序不一定单调，最后统一排序。

    对比 2801：那道题 high 可达 10^100，只能「数个数」；这道题的范围在整数内，
    可以「枚举出来」。计数用 DP，枚举用 BFS，这是同一约束的两种问法。

复杂度：时间 O(答案个数)，空间 O(答案个数)。
"""

from collections import deque


def stepping_numbers(low, high):
    result = []
    if low <= 0 <= high:
        result.append(0)
    q = deque(range(1, 10))
    while q:
        x = q.popleft()
        if x > high:
            continue
        if x >= low:
            result.append(x)
        last = x % 10
        if last > 0:
            q.append(x * 10 + last - 1)
        if last < 9:
            q.append(x * 10 + last + 1)
    result.sort()
    return result


if __name__ == "__main__":
    assert stepping_numbers(0, 21) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 21]
    assert stepping_numbers(10, 15) == [10, 12]
    assert stepping_numbers(100, 100) == []
    assert stepping_numbers(1, 1) == [1]
    print("stepping_numbers: all tests passed")
