class Solution:
    def myAtoi(self, s: str) -> int:
        # Constants for the 32-bit signed integer range
        INT_MIN, INT_MAX = -2 ** 31, 2 ** 31 - 1

        # Step 1: Ignore leading whitespaces
        i = 0
        while i < len(s) and s[i] == ' ':
            i += 1

        # If we reach the end of the string, return 0
        if i == len(s):
            return 0

        # Step 2: Sign detection
        sign = 1
        if s[i] == '-':
            sign = -1
            i += 1
        elif s[i] == '+':
            i += 1

        # Step 3: Convert digits to integer
        result = 0
        while i < len(s) and s[i].isdigit():
            digit = int(s[i])
            # Step 4: Check for overflow and underflow before adding the digit
            if result > (INT_MAX - digit) // 10:
                return INT_MAX if sign == 1 else INT_MIN
            result = result * 10 + digit
            i += 1

        return sign * result
