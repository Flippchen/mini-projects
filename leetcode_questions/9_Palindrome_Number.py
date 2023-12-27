class Solution:
    def isPalindrome(self, x: int) -> bool:
        # A negative number cannot be a palindrome
        if x < 0:
            return False

        # Calculate the reverse of the number
        original, reverse = x, 0
        while x > 0:
            reverse = reverse * 10 + x % 10
            x = x // 10

        # Check if the original number is the same as its reverse
        return original == reverse
