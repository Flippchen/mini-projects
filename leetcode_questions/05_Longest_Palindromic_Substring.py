class Solution:
    def longestPalindrome(self, s: str) -> str:
        def expand_around_center(left: int, right: int) -> str:
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left + 1:right]

        if not s or len(s) < 1:
            return ""

        start, end = 0, 0
        for i in range(len(s)):
            # Check for odd-length palindromes
            palindrome1 = expand_around_center(i, i)
            # Check for even-length palindromes
            palindrome2 = expand_around_center(i, i + 1)

            # Update the longest palindrome found so far
            longest_palindrome = max(palindrome1, palindrome2, key=len)
            if len(longest_palindrome) > end - start:
                start = i - (len(longest_palindrome) - 1) // 2
                end = i + len(longest_palindrome) // 2

        return s[start:end + 1]