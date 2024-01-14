class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        # Recursive helper function
        def dp(i, j):
            # If we have reached the end of both strings, it's a match
            if i >= len(s) and j >= len(p):
                return True
            # If we have reached the end of pattern but not the string, it's not a match
            if j >= len(p):
                return False

            # Check if the current characters match
            match = i < len(s) and (s[i] == p[j] or p[j] == '.')

            # If the next character in the pattern is '*', it can match zero or more of the preceding element
            if j + 1 < len(p) and p[j + 1] == '*':
                # dp(i, j + 2) -> '*' matches zero of the preceding element
                # dp(i + 1, j) -> '*' matches one or more of the preceding element (only if match is True)
                return dp(i, j + 2) or (match and dp(i + 1, j))
            else:
                # If no '*', move to the next character in both strings (only if match is True)
                return match and dp(i + 1, j + 1)

        # Start the recursion with the first characters of the strings
        return dp(0, 0)


