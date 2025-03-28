class Solution(object):
    # def isUgly(self, num):
    #     """
    #     :type num: int
    #     :rtype: bool
    #     """
    #     if num <= 0:
    #         return False
    #     if num <= 6:
    #         return True
    #     while num % 2 == 0:
    #         num //= 2
    #     while num % 3 == 0:
    #         num //= 3
    #     while num % 5 == 0:
    #         num //= 5
    #     if num == 1:
    #         return True
    #     return False
    def isUgly(self, num):
        if num <= 0:
            return False
        divisors = [2, 3, 5]
        for d in divisors:
            while num % d == 0:
                num /= d
        return num == 1

if __name__ == '__main__':
    s = Solution()
    print s.isUgly(-2147483648)

# Modified on 2024-09-01 14:19:55.974324

# Modified on 2024-09-15 22:35:08.617355

# Modified on 2024-09-15 22:40:27.424041

# Modified on 2024-10-06 12:22:15.346350

# Modified on 2024-10-22 18:40:55.541872

# Modified on 2024-11-13 18:05:27.029151

# Modified on 2024-12-20 17:19:47.417846

# Modified on 2025-02-20 23:27:00.753760

# Modified on 2025-03-23 16:38:13.764424
