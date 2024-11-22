class Solution(object):

    def __init__(self):
        self.memo = []
        self.memo.append(0)
        self.memo.append(1)

    def fib(self, N):
        """
        DP with memo
        :type N: int
        :rtype: int
        """
        if N < len(self.memo):
            return self.memo[N]
        for i in range(len(self.memo), N + 1):
            self.memo.append(self.memo[i - 1] + self.memo[i - 2])
        return self.memo[N]

    # def fib(self, N):
    #     """
    #     Recursively, O(n)
    #     :type N: int
    #     :rtype: int
    #     """
    #     if N == 0:
    #         return 0
    #     if N == 1:
    #         return 1
    #     return self.fib(N - 1) + self.fib(N - 2)

# Modified on 2024-09-01 14:19:56.044275

# Modified on 2024-09-15 22:35:08.667109

# Modified on 2024-09-15 22:40:27.467348

# Modified on 2024-10-06 12:22:15.385329

# Modified on 2024-10-22 18:40:55.583125

# Modified on 2024-11-13 18:05:27.072645

# Modified on 2024-12-20 17:19:47.477835
