class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)  # Create a dummy node to handle edge cases more easily
        slow = dummy  # Start slow from dummy to handle edge case when we need to delete the head
        fast = head

        # Move fast n steps ahead
        for _ in range(n):
            fast = fast.next

        # Move both pointers until fast reaches the end
        while fast:
            slow = slow.next
            fast = fast.next

        # Remove the nth node from end
        slow.next = slow.next.next

        # Return the new head, which might be updated if the head was removed
        return dummy.next
