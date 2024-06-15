class Solution:
    def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
        # Create a dummy node to form the merged list
        dummy = ListNode()
        current = dummy

        # While both lists are non-empty, compare the values and append the smaller node
        while list1 and list2:
            if list1.val <= list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next

        # If one of the lists is non-empty, append the rest of the nodes
        if list1:
            current.next = list1
        elif list2:
            current.next = list2

        # The merged list is next to the dummy node
        return dummy.next