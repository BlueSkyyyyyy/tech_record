// 707. 设计链表
// 见 design_linked_list.py 的题目与思路说明。
#include <cassert>
#include <iostream>

struct Node {
    int val;
    Node *prev;
    Node *next;
    explicit Node(int v = 0) : val(v), prev(nullptr), next(nullptr) {}
};

class MyLinkedList {
public:
    MyLinkedList() {
        head_ = new Node();
        tail_ = new Node();
        head_->next = tail_;
        tail_->prev = head_;
        size_ = 0;
    }

    int get(int index) {
        Node *node = nodeAt(index);
        return node ? node->val : -1;
    }

    void addAtHead(int val) { addAtIndex(0, val); }

    void addAtTail(int val) { addAtIndex(size_, val); }

    void addAtIndex(int index, int val) {
        if (index < 0 || index > size_) {
            return;
        }
        Node *nxt = (index == size_) ? tail_ : nodeAt(index);
        Node *prev = nxt->prev;
        Node *node = new Node(val);
        node->prev = prev;
        node->next = nxt;
        prev->next = node;
        nxt->prev = node;
        ++size_;
    }

    void deleteAtIndex(int index) {
        Node *node = nodeAt(index);
        if (!node) {
            return;
        }
        node->prev->next = node->next;
        node->next->prev = node->prev;
        delete node;
        --size_;
    }

private:
    Node *nodeAt(int index) {
        if (index < 0 || index >= size_) {
            return nullptr;
        }
        if (index < size_ / 2) {
            Node *cur = head_->next;
            for (int i = 0; i < index; ++i) {
                cur = cur->next;
            }
            return cur;
        }
        Node *cur = tail_->prev;
        for (int i = 0; i < size_ - 1 - index; ++i) {
            cur = cur->prev;
        }
        return cur;
    }

    Node *head_;
    Node *tail_;
    int size_;
};

int main() {
    MyLinkedList ll;
    ll.addAtHead(1);
    ll.addAtTail(3);
    ll.addAtIndex(1, 2);          // 1 -> 2 -> 3
    assert(ll.get(0) == 1);
    assert(ll.get(1) == 2);
    assert(ll.get(2) == 3);
    assert(ll.get(3) == -1);
    assert(ll.get(-1) == -1);
    ll.deleteAtIndex(1);          // 1 -> 3
    assert(ll.get(1) == 3);
    assert(ll.get(2) == -1);
    ll.addAtTail(4);              // 1 -> 3 -> 4
    assert(ll.get(2) == 4);
    ll.deleteAtIndex(0);          // 3 -> 4
    assert(ll.get(0) == 3);
    ll.deleteAtIndex(5);          // 越界，不删
    assert(ll.get(1) == 4);
    ll.addAtIndex(2, 9);          // 尾插：3 -> 4 -> 9
    assert(ll.get(2) == 9);
    ll.addAtIndex(5, 9);          // 越界，不插
    assert(ll.get(2) == 9);

    std::cout << "design_linked_list: all tests passed\n";
    return 0;
}
