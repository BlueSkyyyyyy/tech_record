"""289. 生命游戏（Game of Life）

题目：给定 m x n 的 0/1 棋盘（1 表示活细胞），按康威生命游戏的规则原地更新：
    - 活细胞周围恰有 2 或 3 个活细胞时继续存活，否则死亡；
    - 死细胞周围恰有 3 个活细胞时复活。
要求原地更新，不能另开一个棋盘。

思路（用第二个二进制位存「下一步状态」）：
    原地更新的难点在于：算某一格的新状态时，不能破坏邻居的旧状态。解决思路是
    把「当前状态」和「下一步状态」放进同一个整数的不同二进制位：
      - 最低位存当前状态（读邻居时只取这一位）；
      - 次低位存下一步状态。
    这样写中间状态时不会覆盖旧状态，邻居之间互不干扰。全部算完后统一右移一位，
    次低位就变成了正式的最低位。

    编码：0b10 表示「现在死、下一步活」；0b01 表示「现在活、下一步死」；
    0b00、0b11 表示状态不变。右移一位后 0b11→1、0b10→1、0b01→0、0b00→0。

复杂度：时间 O(m*n)（每格固定看 8 个邻居），空间 O(1)（原地，不另开棋盘）。
"""


def game_of_life(board):
    m, n = len(board), len(board[0])
    for i in range(m):
        for j in range(n):
            live = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n:
                        live += board[ni][nj] & 1
            if board[i][j] & 1:
                board[i][j] = 0b11 if live in (2, 3) else 0b01
            else:
                board[i][j] = 0b10 if live == 3 else 0b00
    for i in range(m):
        for j in range(n):
            board[i][j] >>= 1


if __name__ == "__main__":
    board = [[0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]
    game_of_life(board)
    assert board == [[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0]]

    board = [[1, 1], [1, 0]]
    game_of_life(board)
    assert board == [[1, 1], [1, 1]]

    board = [[0, 0], [0, 0]]
    game_of_life(board)
    assert board == [[0, 0], [0, 0]]

    board = [[1]]
    game_of_life(board)
    assert board == [[0]]

    print("game_of_life: all tests passed")
