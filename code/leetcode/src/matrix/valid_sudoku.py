"""36. 有效的数独（Valid Sudoku）

题目：给定一个 9x9 的数独盘面，判断它是否有效。只需判断已经填入的数字有没有
违反规则：每一行、每一列、每一个 3x3 宫内，数字 1-9 都不能重复。空格用 '.' 表示。
不需要判断题目是否有解。

思路（三组集合 + 一次遍历）：
    为每一行、每一列、每一个 3x3 宫各维护一个集合。遍历盘面中每个已填数字：
      - 若它已在行集合 rows[i] 中，或列集合 cols[j] 中，或宫集合 boxes[k] 中，
        就违反规则，返回 False；
      - 否则把它同时加入这三个集合。

    宫殿编号这样算：k = (i // 3) * 3 + j // 3。把 9 个 3x3 宫按行优先顺序编号
    0-8，这个式子用「所在行块号 * 3 + 列块号」拼出编号，正好一一对应。

    为什么用集合而不是真的去排序比较：集合的插入与查找都是 O(1)，一次遍历即可，
    边读边查还能第一时间发现冲突；空间上每个集合最多存 9 个字符，可视为 O(1)。
"""


def is_valid_sudoku(board):
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    for i in range(9):
        for j in range(9):
            val = board[i][j]
            if val == ".":
                continue
            k = (i // 3) * 3 + j // 3
            if val in rows[i] or val in cols[j] or val in boxes[k]:
                return False
            rows[i].add(val)
            cols[j].add(val)
            boxes[k].add(val)
    return True


if __name__ == "__main__":
    valid = [
        "53..7....",
        "6..195...",
        ".98....6.",
        "8...6...3",
        "4..8.3..1",
        "7...2...6",
        ".6....28.",
        "...419..5",
        "....8..79",
    ]
    assert is_valid_sudoku(valid)

    invalid = [
        "83..7....",
        "6..195...",
        ".98....6.",
        "8...6...3",
        "4..8.3..1",
        "7...2...6",
        ".6....28.",
        "...419..5",
        "....8..79",
    ]
    assert not is_valid_sudoku(invalid)

    empty = ["........." for _ in range(9)]
    assert is_valid_sudoku(empty)

    print("is_valid_sudoku: all tests passed")
