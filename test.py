class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def swapPairs(self, head):
        if not head or not head.next:
            print('head: {}'.format(head))
            return head
        first = head
        sec = head.next
        first.next = self.swapPairs(sec.next)
        sec.next = first
        print('first: {}, sec: {}, first.next: {}, sec.next: {}'.format(first.val, sec.val, first.next.val, sec.next.val))
        return sec


# 创建链表节点
node5 = ListNode(5)
node4 = ListNode(4, node5)
node3 = ListNode(3, node4)
node2 = ListNode(2, node3)
node1 = ListNode(1, node2)

# 链表的头节点
head = node1

# 实例化Solution类
solution = Solution()

# 调用swapPairs方法进行两两交换
new_head = solution.swapPairs(head)

# 打印交换后的链表
current = new_head
while current:
    print(current.val, end=" -> ")
    current = current.next
print("None")
