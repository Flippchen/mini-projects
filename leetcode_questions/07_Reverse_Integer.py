class Solution:
    def reverse(self, x: int) -> int:
        # Define the range for a 32-bit signed integer
        INT_MIN, INT_MAX = -2**31, 2**31 - 1

        # Initialize result
        result = 0

        # Check if the number is negative
        is_negative = x < 0
        x = abs(x)

        while x != 0:
            # Pop the last digit from x
            pop = x % 10
            x //= 10

            # Check for overflow
            if result > INT_MAX // 10 or (result == INT_MAX // 10 and pop > 7):
                return 0

            # Check for underflow
            if result < INT_MIN // 10 or (result == INT_MIN // 10 and pop > 8):
                return 0

            # Push the digit to the result
            result = result * 10 + pop

        # Return the result with the correct sign
        return -result if is_negative else result


