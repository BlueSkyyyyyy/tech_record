"""720. 词典中最长的单词

题目：给定一个字符串数组 words，找出其中最长的单词，要求该单词可以由 words 中
      的其它单词「每次加一个字母」逐步拼成（也就是它的每个前缀都必须是 words
      里的单词）。若有多个答案，返回字典序最小的那个；没有则返回空串。

思路：先把所有单词插进前缀树，节点用 "#" 标记「到此是一个完整单词」。
      再从根做一次 DFS：只有当某个孩子也是单词（含 "#"）时才继续往下走，
      这样走出来的路径天然满足「每个前缀都是单词」。走到一个单词节点就用
      「更长，或一样长但字典序更小」更新答案。

      为什么能剪掉「前缀不是单词」的分支：题目要求每一步加一个字母后的中间结果
      也要是单词，所以在树上行走时一旦某个孩子没有 "#"，从它再往下的任何路径都
      不可能合法，整枝可以直接砍掉。

复杂度：建树 O(总字符数)；DFS O(总字符数)。空间 O(总字符数)。
"""


def longest_word(words):
    trie = {}
    for word in words:
        node = trie
        for ch in word:
            node = node.setdefault(ch, {})
        node["#"] = True

    best = ""

    def dfs(node, path):
        nonlocal best
        if "#" in node:
            if len(path) > len(best) or (len(path) == len(best) and path < best):
                best = path
        for ch, child in node.items():
            if ch != "#" and "#" in child:
                dfs(child, path + ch)

    dfs(trie, "")
    return best


if __name__ == "__main__":
    words = ["w", "wo", "wor", "worl", "world"]
    assert longest_word(words) == "world"

    words = ["a", "banana", "app", "appl", "ap", "apply", "apple"]
    assert longest_word(words) == "apple"

    # 多个等长答案取字典序最小
    assert longest_word(["ab", "a", "abc", "abd"]) == "abc"
    # 中间某个前缀不是单词时，整条更长的链作废
    assert longest_word(["b", "ba", "ban", "bandanas", "bandana"]) == "ban"
    # 首字母就不在词典里，任何单词都无法起步
    assert longest_word(["xy", "xyz"]) == ""
    print("longest_word: all tests passed")
