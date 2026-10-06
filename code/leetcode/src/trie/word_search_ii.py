"""212. 单词搜索 II

题目：给定一个 m x n 的字符网格 board 和一个单词列表 words，找出所有同时在网格中
      出现的单词。单词必须由相邻单元格（上下左右）的字母依次连接而成，同一单元格
      在一次单词中不能重复使用。

思路：如果对每个单词单独做一遍网格 DFS，代价是「单词数 x 网格大小」，非常浪费。
      把这批单词全部插入一棵前缀树，然后从网格的每个格子出发只做一次 DFS：
      沿着前缀树往下走，一旦当前路径不再是树里某条分支就立刻停下，一旦走到某个
      单词的结尾就把这个词收进答案。

      为什么前缀树能把「多单词搜索」压成「一次遍历」：所有单词共享前缀，网格 DFS
      在某一格选了一个字母后，只要这个字母在某个待搜单词的前缀里，才值得继续；
      否则无论后面怎么走都不可能凑出任何一个单词，当即剪枝。前缀树恰好用 O(1)
      判断「当前前缀是否还是某些单词的前缀」，把大量无效分支提前砍掉。

      为什么要就地修改 board 来去重和防回头：把访问过的格子临时改成一个不在字母表
      里的字符（如空串或 '#'），DFS 回来再还原，相当于用一个访问标记数组，却省下了
      额外空间。同一单词可能通过不同路径被匹配到多次，因此找到后把该节点的结尾标记
      清掉，保证每个单词只收集一次。

复杂度：建树 O(总字符数)。搜索最坏 O(m * n * 4^L)，但对共享前缀的单词集合，
      前缀树会大幅剪枝；空间 O(总字符数 + L)，L 为最长单词长度。
"""


def find_words(board, words):
    trie = {}
    for word in words:
        node = trie
        for ch in word:
            node = node.setdefault(ch, {})
        node["#"] = word

    rows, cols = len(board), len(board[0])
    result = []

    def dfs(r, c, node):
        ch = board[r][c]
        nxt = node.get(ch)
        if nxt is None:
            return
        word = nxt.get("#")
        if word is not None:
            result.append(word)
            nxt["#"] = None  # 置空，避免同一单词被重复收集
        board[r][c] = ""  # 标记已访问
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] in nxt:
                dfs(nr, nc, nxt)
        board[r][c] = ch  # 还原

    for r in range(rows):
        for c in range(cols):
            if board[r][c] in trie:
                dfs(r, c, trie)
    return result


if __name__ == "__main__":
    board = [
        ["o", "a", "a", "n"],
        ["e", "t", "a", "e"],
        ["i", "h", "k", "r"],
        ["i", "f", "l", "v"],
    ]
    words = ["oath", "pea", "eat", "rain"]
    got = sorted(find_words(board, words))
    assert got == ["eat", "oath"]

    board2 = [["a", "a"]]
    assert find_words(board2, ["a"]) == ["a"]
    assert find_words(board2, ["a", "a"]) == ["a"]
    print("word_search_ii: all tests passed")
