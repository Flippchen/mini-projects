class Solution:
    def mySqrt(self, x: int) -> int:
        if x < 2:
            return x

        left, right = 2, x // 2

        while left <= right:
            mid = left + (right - left) // 2
            guessed_square = mid * mid

            if guessed_square == x:
                return mid
            elif guessed_square > x:
                right = mid - 1
            else:
                left = mid + 1

        return right


